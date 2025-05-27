from pathlib import Path
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.interpolate as interpolate
from scipy.signal import medfilt, find_peaks
import pyclipper as pc

from connected_peaks_model.xy import XY
from connected_peaks_model.connected_peaks_model import ConnectedPeaksModel


class Image:
    """Class for a single gantry angle image (displaying gafchromic film)."""

    def __init__(self, path: Path):
        """Initialises GantryAngleImage class by reading image and cropping/rotating to gaf.

        Args:
            path (Path): Path to image.
        """
        self.path = path

        image = cv2.imread(path)
        image, self.cont = self.rotate_and_crop_to_gaf(image)

        self.image = {
            "RGB": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "GRAY": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        }

        print(f"Image loaded: {self.path.name}")

    @classmethod
    def rotate_and_crop_to_gaf(cls, image: np.ndarray) -> np.ndarray:
        pad_x = image.shape[0] // 10
        pad_y = image.shape[1] // 10
        padded = cv2.copyMakeBorder(
            image,
            pad_x,
            pad_x,
            pad_y,
            pad_y,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        # get film contour and its angle
        padded_gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        cont = cls.locate_film(padded_gray)
        _, (w, h), cont_angle = cv2.minAreaRect(cont)
        if w > h:
            cont_angle += 90

        # Calculate the center of the padded image
        (h, w) = padded.shape[:2]
        center = (w // 2, h // 2)

        # Calculate the rotation matrix based on the contour angle
        matrix = cv2.getRotationMatrix2D(center, cont_angle, 1.0)

        # Get the new bounding box dimensions of padded image after rotation
        abs_cos = abs(matrix[0, 0])
        abs_sin = abs(matrix[0, 1])

        # Calculate the new width and height of padded image after rotation
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        # Adjust the rotation matrix to account for translation (shifting the image to prevent clipping)
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]

        # Rotate the image and resize it to the new size
        padded_rotated = cv2.warpAffine(
            padded,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # get rotated contour and bounding box
        cont_rot = cv2.transform(cont, matrix)
        x_c, y_c, w_c, h_c = cv2.boundingRect(cont_rot)
        cont_rot_in_crop_frame = cont_rot - np.array([[x_c, y_c]])

        return (
            padded_rotated[y_c : y_c + h_c, x_c : x_c + w_c],
            cont_rot_in_crop_frame,
        )

    @staticmethod
    def locate_film(image: np.ndarray):
        """Locates the gafchromic film in the image, returning contour enclosing film.

        Args:
            image (np.ndarray): Image to locate film within.

        Returns:
            tuple[list[np.ndarray], np.ndarray]: contour enclosing film.
        """

        gauss_blur = cv2.GaussianBlur(image, (3, 3), 0)
        _, mask = cv2.threshold(gauss_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        conts, _ = cv2.findContours(~mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cont = sorted(conts, key=lambda c: len(c), reverse=True)[0]

        return cont

    def analyse_image(self):
        print(f"Analysing image: {self.path.name}")
        self.cont_inset = self.get_contour_inset()
        self.line_end_points = self.get_line_end_points()
        pass

    def get_contour_inset(self):
        # reshape and upscale contour so can use with pyclipper
        image = self.image["GRAY"]
        cont = self.cont.reshape(-1, 2)
        cont = cont.astype(np.int16)
        pco = pc.PyclipperOffset()
        pco.AddPath(cont, pc.JT_ROUND, pc.ET_CLOSEDPOLYGON)
        abs_shift = min(image.shape) // 10
        cont_inset = np.array(pco.Execute(-abs_shift), dtype=np.int64).reshape(-1, 1, 2)
        cont_inset = self.resample_with_equidistance(cont_inset, num_points=2000)

        return cont_inset

    @staticmethod
    def resample_with_equidistance(contour, num_points: int):
        points = contour[:, 0, :]

        # Calculate cumulative distance along contour as an array
        diffs = np.diff(points, axis=0, append=[points[0]])
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        cum_dists = np.insert(np.cumsum(dists), 0, 0)

        # Create x and y interpolators in domain of cumulative distances
        interp_x = interpolate.interp1d(
            cum_dists, np.append(points[:, 0], points[0, 0]), kind="linear"
        )
        interp_y = interpolate.interp1d(
            cum_dists, np.append(points[:, 1], points[0, 1]), kind="linear"
        )

        # Perform uniform resampling
        even_dists = np.linspace(0, cum_dists[-1], num_points)
        resampled = (
            np.column_stack((interp_x(even_dists), interp_y(even_dists)))
            .reshape(-1, 1, 2)
            .astype(int)
        )
        return resampled

    def get_line_end_points(self):
        self.signal, roll = self.get_signal()
        cpm = ConnectedPeaksModel(self.signal)
        crossing_idxs = cpm.get_crossing_indices()
        unrolled_idxs = [(idx + roll) % len(self.cont_inset) for idx in crossing_idxs]
        contour_pts = self.cont_inset[unrolled_idxs, :, :].reshape(-1, 2)
        return contour_pts

    def get_signal(self) -> XY:
        # Get signal from inset contour
        signal = self.image["GRAY"].astype(float)[
            self.cont_inset[:, 0, 1], self.cont_inset[:, 0, 0]
        ]

        # Invert and roll signal to global minimum
        k = len(signal) // 30 + (len(signal) // 30 + 1) % 2
        signal = medfilt(signal, k)
        signal *= -1
        signal -= np.min(signal)
        roll = np.argmin(signal)
        signal = np.roll(signal, -roll)
        signal = XY(np.arange(len(signal)), signal)

        return signal, roll
