"""
gantry_angle_image.py

Script defining the GantryAngleImage class, which contains all gantry
angle analysis at the iamge level. Preprocesses the image when initialised.
When analysis function is called, the angles between the irradiated slices are
calculated and a figure is displayed and saved.

Written by Nathan Crossley, 2025.
"""

from pathlib import Path

import numpy as np
import scipy.interpolate as interpolate
from scipy.signal import medfilt, find_peaks

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

import cv2
import pyclipper as pc

from src.image_processing.filters import bilateral_filter_1d
from src.image_processing.xy import XY
from src.pipeline.io_manager import IOManager


class GantryAngleImage:
    """Class for an image for the gantry angle task.
    This should be an image of gafchromic film, clearly
    showing three crossing irradiated CT slices."""

    def __init__(self, path: Path):
        """Initialises GantryAngleImage class by reading image
        and cropping/rotating to gaf. RGB and grayscale representations are stored.

        Args:
            path (Path): Path to image.
        """

        # store file path to image
        self.path = path

        # read image using OpenCV in BGR format
        image = cv2.imread(path)

        # rotate the image so gaf is axis aligned and then crop
        image, self.contour = self._rotate_and_crop_to_gaf(image)

        # store RGB and grayscale representations of image
        self.image = {
            "RGB": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "GRAY": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        }

        # calculate wedge radius for particular image (used later for plots)
        self.wedge_radius = min(self.image["GRAY"].shape) // 6

        # inform user that the image has been loaded
        print(f"Image loaded: {self.path.name}")

    @classmethod
    def _rotate_and_crop_to_gaf(
        cls, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Takes a BGR image of gafchromic film and rotates so the film
        is axis aligned. Also crops the image to the film.

        Args:
            image (np.ndarray): _BGR image to crop and rotate.

        Returns:
           tuple[np.ndarray, np.ndarray]: Cropped and rotated image and gaf film contour in that frame.
        """

        # pad the image with a border to help with contour detection.
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

        # get grayscale representation of the image
        padded_gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)

        # locate contour around film and get angle, width and height of minAreaRect
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
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        # Adjust the rotation matrix to account for translation (shifting the image to prevent clipping)
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]

        # Apply affine matrix to rotate with no clip
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
    def _locate_film(image: np.ndarray) -> np.ndarray:
        """Locates the gafchromic film in the image,
        returning contour enclosing film.

        Args:
            image (np.ndarray): Image to locate film within.

        Returns:
            np.ndarray: contour enclosing film.
        """

        # apply Gaussian blur to remove noise from image.
        gauss_blur = cv2.GaussianBlur(image, (3, 3), 0)

        # binarise image into mask using Otsu threshold (good enough to pick out film edge)
        _, mask = cv2.threshold(gauss_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # find contours and sort by length to get longest contour
        conts, _ = cv2.findContours(~mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cont = sorted(conts, key=lambda c: len(c), reverse=True)[0]

        # Enforce counter-clockwise direction on contour
        if cv2.contourArea(cont, oriented=True) < 0:
            cont = cont[::-1]

        return cont

    def analyse(self):
        """Main function to trigger analysis process for
        this particular image. Insets the film contour, takes
        a line profile across this inset contour and detects points
        where contour crosses across irradiated slices. Based on these
        points, lines are formed to represent the edges of the three slices.
        Lines are paired up and angles found between pairs to get angles between
        irradiated slices.
        """

        # try pipeline analysis
        try:

            # get test contour for detecting edges of imaged slices (an inset contour)
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

            # form "lines" on the image that represent the detected edges of the irradiated slices by pairng up points
            self.lines = self._form_lines()

            # pair up lines into groups of two - lines paired such that the angles between lines represent the angles between slices
            self.paired_lines = self._pair_up_lines()

            # calculate the angles of individual lines in paired lines wrt +ve x axis
            self._calc_angles_wrt_x()

            # calculate necessary angles between irradiated slices and display then save figure
            self._calc_angles()

            # inform user that this image has been analysed.
            print(f"{self.path.name}: Analysed image.")

        # if exception raised, inform user and skip image
        except Exception as e:
            print(f"{self.path.name}: Could not analyse image because of error: {e}")

    def _inset_gaf_contour(self) -> np.ndarray:
        """Returns an inset version of the contour of the
        film edge. Absolute shift inwards depends on size of image.
        This is useful to avoid edge effects when taking line profile/
        signal across contour.

        Returns:
            np.ndarray: Inset contour.
        """
        # get grayscale image, reshape film contour and get as int type
        image = self.image["GRAY"]
        cont = self.contour.reshape(-1, 2)
        cont = cont.astype(np.int16)

        # use pyclipper to inset contour by absolute shift (defined as 1/30 of min image dim) and reshape expected contour shape for cv2
        pco = pc.PyclipperOffset()
        pco.AddPath(cont, pc.JT_ROUND, pc.ET_CLOSEDPOLYGON)
        abs_shift = min(image.shape) // 30
        inset_contour = np.array(pco.Execute(-abs_shift), dtype=np.int64).reshape(
            -1, 1, 2
        )

        # resample inset contour so contains a certain number of equal spaced points
        inset_contour = self._resample_with_equidistance(inset_contour, num_points=2000)

        return inset_contour

    @staticmethod
    def _resample_with_equidistance(contour: np.ndarray, num_points: int) -> np.ndarray:
        """Resample a contour to contain a certain number of points,
        each equally spaced around the contour.

        Args:
            contour (np.ndarray): Contour to resample.
            num_points (int): Number of points wanted in resampled contour.

        Returns:
            np.ndarray: Original contour resampeld with num_points equally spaced points
        """

        # extract contour points
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

    def _get_slice_edge_points(self) -> tuple[np.ndarray]:
        """Calculates the coordinates of "slice edge points",
        which are the spatial points on the test contour that correspond
        to the edges of the irradiated slices. Various filtered signals/line
        profiles are returned for convenience.

        Returns:
            tuple[np.ndarray]: Slice edge points, original signal,
                smoothed signal, derivative of signal and amplified derivative.
        """

        # get contrast enhanced version of grayscale image by gaussian blur and CLAHE.
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

        # smooth signal with bilateral filter with spatial and color sigmas determined by profile properties.
        sigma_space = len(signal.x) // 100
        sigma_color = np.ptp(signal.y) / 5
        signal_smoothed = bilateral_filter_1d(signal.y, sigma_color, sigma_space)
        signal_smoothed = XY(signal.x, signal_smoothed)

        # get gradient of smoothed signal
        signal_deriv = XY(
            signal_smoothed.x, np.gradient(signal_smoothed.y, signal_smoothed.x)
        )

        # get an 'amplified derivative', which is the product of the derivative and the smoothed signal.
        # this is useful in following peak detection as want to give preferential weighting to high gradients that have
        # large y in original signal
        signal_amplified_deriv = XY(signal_deriv.x, signal_deriv.y * signal_smoothed.y)

        # get idxs of six highest gradients in each direction (positive and negative), combine into one list.
        six_highest_gradients = self._get_N_highest_peaks(signal_amplified_deriv.y, 6)
        six_lowest_gradients = self._get_N_highest_peaks(-signal_amplified_deriv.y, 6)
        crossing_idxs = six_highest_gradients + six_lowest_gradients

        # undo rolling from earlier and map to physical points on test contour, ensuring that idxs are in ascending order
        unrolled_idxs = [
            (idx + future_roll) % len(self.test_contour) for idx in crossing_idxs
        ]
        unrolled_idxs = sorted(unrolled_idxs)
        slice_edge_points = self.test_contour[unrolled_idxs, :, :].reshape(-1, 2)

        # throw error if number of slice edge points not as expected as analysis will be inaccurate
        if len(slice_edge_points) != 12:
            raise ValueError(
                f"Number of slice edge points detected differs from what is expected ({len(slice_edge_points)} not 12). Results will be inaccurate."
            )

        # order slice edge points by rolling array so that edge point at bottom left of image is first in array
        # this is required when edge points are paired to form lines.
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
        """Get basic signal/line profile across contour using image as reference
        for signal intensity.

        Args:
            image (np.ndarray): Image to use as reference for signal intensity
            contour (np.ndarray): contour across which to take signal intensity

        Returns:
            tuple[XY, int]: signal and roll required to get minimum of signal at idx=0
        """

        # get signal by slicing with contour coords.
        signal = image.astype(float)[contour[:, 0, 1], contour[:, 0, 0]]

        # apply median filter to get rid of harsh noise
        k = len(signal) // 30 + (len(signal) // 30 + 1) % 2
        signal = medfilt(signal, k)

        # invert signal so slices become maxima rather than minima (more intuitive)
        signal *= -1
        signal -= np.min(signal)

        # roll signal so global min is at idx 0
        # this stops a peak from potentially spanning from signal end -> signal start
        # which would not be useful for later peak detection.
        roll = np.argmin(signal)
        signal = np.roll(signal, -roll)
        signal = XY(np.arange(len(signal)), signal)

        return signal, roll

    @staticmethod
    def _get_N_highest_peaks(
        signal: np.ndarray, N: int, padding: int = 5
    ) -> np.ndarray:
        """Returns the N highest peaks in the passed signal.
        A padding can be passed to apply a linear ramp to signal edges,
        to allow high values at signal edges to pass as peaks.

        Args:
            signal (np.ndarray): Signal for peak detection.
            N (int): Number of peaks to find.
            padding (int, optional): How much padding to apply for linear ramp. Defaults to 5.

        Returns:
            np.ndarray: The idxs for the N highest peaks in the signal
        """

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

    def _roll_slice_edge_points(self, slice_edge_points: np.ndarray) -> np.ndarray:
        """Roll the slice edge points such that the edge point in the bottom
        left corner of the image is first in the array.

        Args:
            slice_edge_points (np.ndarray): List of slice edge points

        Returns:
            np.ndarray: List of slice edge points, post rolling
        """

        # lsm stands for left side midpoint
        # get an approimation for lsm
        centre_y = np.mean(self.test_contour, axis=0)[0][1]
        x_min = np.min(self.test_contour[:, 0, 0])
        lsm_approx = np.array([[x_min, centre_y]])

        # find the point on the test contour that is closest to the lsm
        diff = self.test_contour - lsm_approx
        norms = np.linalg.norm(diff[:, 0, :], axis=1)
        lsm_true_idx = np.argmin(norms)

        # iterate around contour clockwise until a point is found that is in self.slice_edge_points
        # store the corresponding idx in self.slice_edge_poitns
        # this point will be necessary to know as a starting point for pairing up slice edge points
        i = lsm_true_idx - 1
        while i != lsm_true_idx:
            test_point = self.test_contour[i, 0, :]
            bool_check = [np.array_equal(test_point, lep) for lep in slice_edge_points]
            if any(bool_check):
                break
            i = (i - 1) % len(self.test_contour)
        target_idx = np.where(bool_check)[0][0]

        # roll slice edge points by idx to get bottom left point at start of array
        rolled_slice_edge_points = np.roll(slice_edge_points, -target_idx, axis=0)

        return rolled_slice_edge_points

    def _form_lines(self) -> list[dict]:
        """Form lines representing the edges of the slices on the images,
        by systematically pairing up slice edge points.

        Returns:
            list[dict]: List of lines, each represented by a dict containing
                its start and end points.
        """

        # init empty list to receive lines info
        lines = []

        # iterate over each edge point
        for i, point_selected in enumerate(self.slice_edge_points):

            # pair up point by taking (i+5)th or (i+7)th point in array, depending on whether i is odd or even.
            # this relationship is derived from examining the expected structure of test images
            point_to_pair = self.slice_edge_points[
                (i + (5 if i % 2 == 0 else 7)) % len(self.slice_edge_points)
            ]

            # only append line to list, if line already exists with start and end points reversed (this is just a copy)
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

    def _pair_up_lines(self) -> dict[dict[dict]]:
        """Arrange lines into pairs such that the angles between
        lines will give the angles between irradiated slices. Pairing
        has been determined by examining test images are should be consistent
        as slice edge points were rolled to always get bottom left point
        as the first in the array.

        Returns:
            dict[dict[dict]]: Nested dict documenting pairs of lines. Each line
                is, as before, represented by a dict of start and end points.
        """

        # define line idxs for pairing, with the label being descriptive of where
        # where in the image the angle between the lines lies.
        line_idxs_and_labels = [
            [2, 3, "top_left"],
            [0, 4, "top_right"],
            [3, 5, "bottom_left"],
            [1, 4, "bottom_right"],
        ]

        # pair up lines into nested dict
        paired_lines = {
            label: {"line_1": self.lines[i], "line_2": self.lines[j]}
            for (i, j, label) in line_idxs_and_labels
        }

        return paired_lines

    def _calc_angles_wrt_x(self):
        """Calculates the angles of all lines in paired_lines
        dict wrt the +ve x axis, in radians.
        """

        # iterate over all lines present in self.paired_lines, calculate angle
        # of line wrt +ve x axis and append as an attribute in line dict.
        for line_pair in self.paired_lines.values():
            for line_info in line_pair.values():
                start_point = line_info["start_point"]
                end_point = line_info["end_point"]
                angle_wrt_x = np.arctan2(
                    end_point[1] - start_point[1], end_point[0] - start_point[0]
                )
                line_info["angle_wrt_x"] = angle_wrt_x % (2 * np.pi)

    def _calc_angles(self):
        """Calculates the angles between all paired lines and adds each
        to the paired_line dict.
        """

        # init fig and axes for plotting
        fig, ax = plt.subplots()

        # show RGB image of gaf film
        ax.imshow(self.image["RGB"])

        # iterate over all line pairs individually
        for i, line_pair in enumerate(self.paired_lines.values()):

            # extract lines 1 and 2 for convenience
            line_1 = line_pair["line_1"]
            line_2 = line_pair["line_2"]

            # find the intersection between lines 1 and 2
            intersection = self._find_line_intersection(
                line_1["start_point"],
                line_1["end_point"],
                line_2["start_point"],
                line_2["end_point"],
            )

            # get the angles of lines 1 and 2 wrt +ve x axis
            angle_1 = line_1["angle_wrt_x"]
            angle_2 = line_2["angle_wrt_x"]

            # correct angles so that they can be used in Wedge display
            angle_1, angle_2 = self._adjust_angles_for_display(
                angle_1, angle_2, intersection
            )

            # calculate intersection angle between lines and add to relevant line pair dict
            line_pair["intersection_angle"] = np.round(
                np.degrees((angle_2 - angle_1) % (2 * np.pi)), 2
            )

            # draw a wedge on the plot, representing the angle between the two lines
            wedge = Wedge(
                center=intersection,
                r=self.wedge_radius,
                theta1=np.degrees(angle_1),
                theta2=np.degrees(angle_2),
                color=f"C{i}",
                label=f"{line_pair['intersection_angle']}°",
            )
            ax.add_patch(wedge)

            # draw faint lines representing detected slice edges
            # helps the user to see whether the edges have been detected properly
            for line in [line_1, line_2]:
                ax.plot(
                    [line["start_point"][0], line["end_point"][0]],
                    [line["start_point"][1], line["end_point"][1]],
                    color="blue",
                    alpha=0.15,
                )

            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

        # save figure using path taken from IOManager
        plt.savefig(
            IOManager.paths_from_proj_dir["figures_dir"].joinpath(
                f"{self.path.stem}_results.png"
            ),
            dpi=600,
        )

        # show figure and wait for user to close.
        plt.show(block=True)
        plt.close()

    @staticmethod
    def _find_line_intersection(
        start_point_1: np.ndarray,
        end_point_1: np.ndarray,
        start_point_2: np.ndarray,
        end_point_2: np.ndarray,
    ) -> np.ndarray:
        """Finds the intersection between two lines, given both of their start
        and end points.

        Args:
            start_point_1 (np.ndarray): Start point for line 1.
            end_point_1 (np.ndarray): End point for line 1.
            start_point_2 (np.ndarray): Start point for line 2.
            end_point_2 (np.ndarray): End point for line 2.

        Returns:
            np.ndarray: Intersection point of lines 1 and 2.
        """

        # ensure that start and end points are np.ndarray as required for vectorised operations
        start_point_1, end_point_1, start_point_2, end_point_2 = map(
            np.array, [start_point_1, end_point_1, start_point_2, end_point_2]
        )

        # find displacement vectors for lines 1 and 2
        d1 = end_point_1 - start_point_1
        d2 = end_point_2 - start_point_2

        # find the determinant of displacement matrix
        determinant = d1[0] * d2[1] - d1[1] * d2[0]

        # if determinant is zero, liens don't intersect
        if np.isclose(determinant, 0):
            return None

        # find value of parametric parameter t, that represents distance along line
        # to get to intersection point.
        t = (
            (start_point_2[0] - start_point_1[0]) * d2[1]
            - (start_point_2[1] - start_point_1[1]) * d2[0]
        ) / determinant

        # calcualte intersection
        intersection = start_point_1 + t * d1

        return intersection

    def _adjust_angles_for_display(
        self, angle_1: float, angle_2: float, intersection: np.ndarray
    ):
        """Adjusts a

        Args:
            angle_1 (float): _description_
            angle_2 (float): _description_
            intersection (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        # Swap and normalise angles 1 and 2 so that angle between 1 and 2 is the acute angle
        if (angle_2 - angle_1) % (2 * np.pi) > np.pi / 2:
            angle_1, angle_2 = angle_2, (angle_1 + np.pi) % (2 * np.pi)

        # get the midpoint of the arc between the arc endpoints for angle wedge
        mean_angle = ((angle_1 + angle_2) % (2 * np.pi)) / 2
        arc_midpoint = intersection + self.wedge_radius * np.array(
            [np.cos(mean_angle), np.sin(mean_angle)]
        )

        # get centre of test contour
        test_contour_centre = np.mean(self.test_contour, axis=0)[0]

        # check whether reflecting the arc midpoint about the intersection point
        # causes it to move away from the test contour centre
        if self._reflection_further_to_reference(
            arc_midpoint, intersection, test_contour_centre
        ):
            # if so, reflect wedge and normalise angles
            angle_1 += np.pi
            angle_2 += np.pi
            angle_1, angle_2 = angle_1 % (2 * np.pi), angle_2 % (2 * np.pi)

        return angle_1, angle_2

    @staticmethod
    def _reflection_further_to_reference(
        point: np.ndarray, reflection_centre: np.ndarray, reference_point: np.ndarray
    ) -> bool:
        """Reflects a point 180 across a reflection centre point.
        Checks to see if that point is closer to a reference point.

        Args:
            point (np.ndarray): Point to reflect across centre
            reflection_centre (np.ndarray): Centre of reflection
            reference_point (np.ndarray): Reference point for post-reflection validation.

        Returns:
            bool: True if reflected point is closer to reference point, False otherwise.
        """
        reflected = 2 * reflection_centre - point
        return np.linalg.norm(reflected - reference_point) > np.linalg.norm(
            point - reference_point
        )
