from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import cv2
import scipy.interpolate as interpolate
from scipy.signal import medfilt, find_peaks
import pyclipper as pc

from filters import bilateral_filter_1d
from xy import XY
from io_manager import IOManager


class Image:
    """Class for a single gantry angle image (displaying gafchromic film)."""

    def __init__(self, path: Path):
        """Initialises GantryAngleImage class by reading image and cropping/rotating to gaf.

        Args:
            path (Path): Path to image.
        """
        self.path = path

        image = cv2.imread(path)
        image, self.contour = self._rotate_and_crop_to_gaf(image)

        self.image = {
            "RGB": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "GRAY": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        }
        self.wedge_radius = min(self.image["GRAY"].shape) // 6

        print(f"Image loaded: {self.path.name}")

    @classmethod
    def _rotate_and_crop_to_gaf(cls, image: np.ndarray) -> np.ndarray:
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
        cont = cls._locate_film(padded_gray)
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
    def _locate_film(image: np.ndarray):
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

        # Enforce counter-clockwise direction on contour
        if cv2.contourArea(cont, oriented=True) < 0:
            cont = cont[::-1]

        return cont

    def analyse(self):
        try:
            # get test contour for detecting edges of imaged slices
            self.test_contour = self._inset_gaf_contour()

            # get "slice edge points" i.e. points on a contour that define the edges of the imaged slices
            # store used contours and profiles for convenience
            (
                self.slice_edge_points,
                self.signal,
                self.signal_smoothed,
                self.signal_deriv,
                self.signal_amplified_deriv,
            ) = self._get_slice_edge_points()

            self.lines = self._form_lines()
            self.paired_lines = self._pair_up_lines()
            self._calc_angles_wrt_x()
            self._calc_angles()

            print(f"{self.path.name}: Analysed image.")

        except Exception as e:
            print(f"{self.path.name}: Could not analyse image because of error: {e}")

    def _inset_gaf_contour(self):
        # reshape and upscale contour so can use with pyclipper
        image = self.image["GRAY"]
        cont = self.contour.reshape(-1, 2)
        cont = cont.astype(np.int16)
        pco = pc.PyclipperOffset()
        pco.AddPath(cont, pc.JT_ROUND, pc.ET_CLOSEDPOLYGON)
        abs_shift = min(image.shape) // 30
        inset_contour = np.array(pco.Execute(-abs_shift), dtype=np.int64).reshape(
            -1, 1, 2
        )
        inset_contour = self._resample_with_equidistance(inset_contour, num_points=2000)

        return inset_contour

    @staticmethod
    def _resample_with_equidistance(contour, num_points: int):
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

    def _get_slice_edge_points(self):

        # get contrast enhanced version of grayscale image
        k_size = max(3, min(self.image["GRAY"].shape) // 50)
        k_size += (k_size + 1) % 2
        image_blurred = cv2.GaussianBlur(self.image["GRAY"], (k_size, k_size), 0)
        image_contrast_enhanced = cv2.createCLAHE(
            clipLimit=1, tileGridSize=(8, 8)
        ).apply(image_blurred)

        # get basic pixel signal from contrast enhanced image and test contour
        signal, future_roll = self._get_basic_signal(
            image_contrast_enhanced, self.test_contour
        )

        # smooth signal with savgol, preventing edge effects with padding (and reverting)
        sigma_space = len(signal.x) // 100
        sigma_color = np.ptp(signal.y) / 5
        signal_smoothed = bilateral_filter_1d(signal.y, sigma_color, sigma_space)
        signal_smoothed = XY(signal.x, signal_smoothed)

        # get gradient of smoothed signal
        signal_deriv = XY(
            signal_smoothed.x, np.gradient(signal_smoothed.y, signal_smoothed.x)
        )
        signal_amplified_deriv = XY(signal_deriv.x, signal_deriv.y * signal_smoothed.y)

        # get six highest gradients in each direction, combine into one list.
        six_highest_gradients = self._get_N_highest_peaks(signal_amplified_deriv.y, 6)
        six_lowest_gradients = self._get_N_highest_peaks(-signal_amplified_deriv.y, 6)
        crossing_idxs = six_highest_gradients + six_lowest_gradients

        # undo rolling from earlier and map to physical points on test contour
        unrolled_idxs = [
            (idx + future_roll) % len(self.test_contour) for idx in crossing_idxs
        ]
        unrolled_idxs = sorted(unrolled_idxs)
        slice_edge_points = self.test_contour[unrolled_idxs, :, :].reshape(-1, 2)

        # throw error if number of slice edge points not as expected

        if len(slice_edge_points) != 12:
            raise ValueError(
                f"Number of slice edge points detected differs from what is expected ({len(slice_edge_points)} not 12). Results will be inaccurate."
            )

        slice_edge_points_rolled = self._roll_slice_edge_points(slice_edge_points)

        return (
            slice_edge_points_rolled,
            signal,
            signal_smoothed,
            signal_deriv,
            signal_amplified_deriv,
        )

    @staticmethod
    def _get_basic_signal(image: np.ndarray, contour: np.ndarray) -> tuple[XY, int]:
        signal = image.astype(float)[contour[:, 0, 1], contour[:, 0, 0]]

        # Invert and roll signal to global minimum
        k = len(signal) // 30 + (len(signal) // 30 + 1) % 2
        signal = medfilt(signal, k)
        signal *= -1
        signal -= np.min(signal)
        roll = np.argmin(signal)
        signal = np.roll(signal, -roll)
        signal = XY(np.arange(len(signal)), signal)

        return signal, roll

    @staticmethod
    def _get_N_highest_peaks(signal, N: int, padding: int = 5):

        # store peak to peak of signal for convenience
        ptp = np.ptp(signal)

        # pad signal with a linear ramp to allow peak detection at edges if relevant
        signal_padded = np.pad(
            signal,
            padding,
            mode="linear_ramp",
            end_values=(signal[0] - ptp * 0.1, signal[-1] - ptp * 0.1),
        )

        # find peaks of padded signal
        peaks, props = find_peaks(
            signal_padded,
            height=np.min(signal_padded),
            prominence=np.ptp(signal_padded) // 20,
            distance=25,
        )

        # sort peaks by height in descending order and undo padding to get original indices
        peaks_sorted = peaks[np.argsort(props["peak_heights"])[::-1]]
        peaks_corrected = [
            p - padding for p in peaks_sorted if 0 <= p - padding < len(signal)
        ]

        return peaks_corrected[:N]

    def _roll_slice_edge_points(self, slice_edge_points):
        # lsm stands for left side midpoint
        # find contour index of point closest to lsm approximation
        centre_y = np.mean(self.test_contour, axis=0)[0][1]
        x_min = np.min(self.test_contour[:, 0, 0])
        lsm_approx = np.array([[x_min, centre_y]])
        diff = self.test_contour - lsm_approx
        norms = np.linalg.norm(diff[:, 0, :], axis=1)
        lsm_true_idx = np.argmin(norms)

        # iterate around contour clockwise until a point is found that is in self.line_end_points
        # this point will be necessary to know as a starting point for pairing up line endpoints
        i = lsm_true_idx - 1
        while i != lsm_true_idx:
            test_point = self.test_contour[i, 0, :]
            bool_check = [np.array_equal(test_point, lep) for lep in slice_edge_points]
            if any(bool_check):
                break
            i = (i - 1) % len(self.test_contour)

        target_idx = np.where(bool_check)[0][0]
        rolled_slice_edge_points = np.roll(slice_edge_points, -target_idx, axis=0)

        return rolled_slice_edge_points

    def _form_lines(self):
        lines = []
        for i, point_selected in enumerate(self.slice_edge_points):
            point_to_pair = self.slice_edge_points[
                (i + (5 if i % 2 == 0 else 7)) % len(self.slice_edge_points)
            ]

            if not any(
                [
                    np.array_equal(
                        [existing_line["start_point"], existing_line["end_point"]],
                        [point_to_pair, point_selected],
                    )
                    for existing_line in lines
                ]
            ):
                lines.append(
                    {"start_point": point_selected, "end_point": point_to_pair}
                )
        return lines

    def _pair_up_lines(self):
        line_idxs_and_labels = [
            [2, 3, "top_left"],
            [0, 4, "top_right"],
            [3, 5, "bottom_left"],
            [1, 4, "bottom_right"],
        ]

        paired_lines = {
            label: {"line_1": self.lines[i], "line_2": self.lines[j]}
            for (i, j, label) in line_idxs_and_labels
        }

        return paired_lines

    def _calc_angles_wrt_x(self):
        for line_pair in self.paired_lines.values():
            for line_info in line_pair.values():
                start_point = line_info["start_point"]
                end_point = line_info["end_point"]
                angle_wrt_x = np.arctan2(
                    end_point[1] - start_point[1], end_point[0] - start_point[0]
                )
                line_info["angle_wrt_x"] = angle_wrt_x % (2 * np.pi)

    def _calc_angles(self):
        fig, ax = plt.subplots()

        ax.imshow(self.image["RGB"])

        for i, line_pair in enumerate(self.paired_lines.values()):
            line_1 = line_pair["line_1"]
            line_2 = line_pair["line_2"]

            intersection = self._find_line_intersection(
                line_1["start_point"],
                line_1["end_point"],
                line_2["start_point"],
                line_2["end_point"],
            )

            angle_1 = line_1["angle_wrt_x"]
            angle_2 = line_2["angle_wrt_x"]

            angle_1, angle_2 = self._adjust_angles_for_display(
                angle_1, angle_2, intersection
            )

            line_pair["intersection_angle"] = np.round(
                np.degrees((angle_2 - angle_1) % (2 * np.pi)), 2
            )

            wedge = Wedge(
                center=intersection,
                r=self.wedge_radius,
                theta1=np.degrees(angle_1),
                theta2=np.degrees(angle_2),
                color=f"C{i}",
                label=f"{line_pair['intersection_angle']}°",
            )
            ax.add_patch(wedge)

            for line in [line_1, line_2]:
                ax.plot(
                    [line["start_point"][0], line["end_point"][0]],
                    [line["start_point"][1], line["end_point"][1]],
                    color="blue",
                    alpha=0.15,
                )

            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

        plt.savefig(
            IOManager.paths_from_proj_dir["figures_dir"].joinpath(
                f"{self.path.stem}_results.png"
            ),
            dpi=600,
        )
        plt.show(block=True)
        plt.close()

    @staticmethod
    def _find_line_intersection(start_point_1, end_point_1, start_point_2, end_point_2):

        start_point_1, end_point_1, start_point_2, end_point_2 = map(
            np.array, [start_point_1, end_point_1, start_point_2, end_point_2]
        )

        d1 = end_point_1 - start_point_1
        d2 = end_point_2 - start_point_2
        determinant = d1[0] * d2[1] - d1[1] * d2[0]

        if np.isclose(determinant, 0):
            return None

        t = (
            (start_point_2[0] - start_point_1[0]) * d2[1]
            - (start_point_2[1] - start_point_1[1]) * d2[0]
        ) / determinant
        intersection = start_point_1 + t * d1
        return intersection

    def _adjust_angles_for_display(self, angle_1, angle_2, intersection):
        # select acute angle
        if (angle_2 - angle_1) % (2 * np.pi) > np.pi / 2:
            angle_1, angle_2 = angle_2, (angle_1 + np.pi) % (2 * np.pi)

        mean_angle = ((angle_1 + angle_2) % (2 * np.pi)) / 2
        arc_midpoint = intersection + self.wedge_radius * np.array(
            [np.cos(mean_angle), np.sin(mean_angle)]
        )

        test_contour_centre = np.mean(self.test_contour, axis=0)[0]
        if self._reflection_further_to_reference(
            arc_midpoint, intersection, test_contour_centre
        ):
            angle_1 += np.pi
            angle_2 += np.pi
            angle_1, angle_2 = angle_1 % (2 * np.pi), angle_2 % (2 * np.pi)

        return angle_1, angle_2

    @staticmethod
    def _reflection_further_to_reference(point, reflection_centre, reference_point):
        """Reflects a point across a reflection centre."""
        reflected = 2 * reflection_centre - point
        return np.linalg.norm(reflected - reference_point) > np.linalg.norm(
            point - reference_point
        )
