"""
xy.py

Script defining a subclass of np.ndarray, XY, for handling linked
1D-arrays of x and y values. This makes later code cleaner and more readable.

Written by Nathan Crossley, 2025.
"""

import numpy as np
from typing import Self
from scipy.signal import find_peaks


class XY(np.ndarray):
    """Subclass of np.ndarray for handling linked 1D-arrays of x and y values.
    Allows for easy manipulation and access of x and y values."""

    def __new__(cls, *args: list[int | float]):
        """Initialise class, checking that input arrays are valid."""

        # check that all input arrays have the same length
        if len(set(map(len, args))) != 1:
            raise ValueError("All input arrays must have the same length")

        # create array from args
        arr = np.array(args)

        # if dimensions are not correct, raise error
        if arr.ndim != 2 or arr.shape[0] != 2:
            raise ValueError("args of XY should be two 1d arrays")

        # return the array as an instance of the class
        return np.asarray(arr).view(cls)

    @property
    def x(self) -> Self:
        """Property for x array"""
        return self[0].flatten()

    @property
    def y(self) -> Self:
        """Property for y array"""
        return self[1].flatten()

    @y.setter
    def y(self, val: np.ndarray):
        """Setter for y property"""

        # check that input is either a list or numpy.ndarray
        if isinstance(val, (np.ndarray, list)):

            # check that input has the same length as current y as should not modify otherwise.
            if len(val) != len(self.y):
                raise ValueError("Cannot modify shape of XY.y")

        # else raise error
        else:
            raise TypeError("Expected input to be either a list or numpy.ndarray")

        # set value of y if passed checks
        self[1] = val

    def find_highest_peak(self) -> tuple[int, int | float]:
        """Finds the index and x-value of the highest peak in the y-array.

        Returns:
            tuple[int, int | float]: Tuple containing the index of the highest peak and its corresponding x-value.
        """

        # get global peak
        peaks, props = find_peaks(self.y, height=0)

        # get index of global peak and corresponding x value
        idx_peak = peaks[np.argmax(props["peak_heights"])]
        x_peak = self.x[idx_peak]

        return idx_peak, x_peak

    def get_union_x_range(self, other_xy) -> np.ndarray:
        """Gets Union (added) x-range from this XY object and another.

        Args:
            other_xy (XY): Other XY object to compare with.

        Returns:
            np.ndarray: array of Union x-range of self and other_xy.
        """

        # get min and max x values from the two objects
        x_min = min(self.x.min(), other_xy.x.min())
        x_max = max(self.x.max(), other_xy.x.max())

        # construct union x range
        x_range = np.arange(x_min, x_max + 1)

        return x_range
