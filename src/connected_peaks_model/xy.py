import numpy as np
from typing import Self
from scipy.signal import find_peaks


class XY(np.ndarray):
    """Class for 2D numpy array for plotting"""

    def __new__(cls, *args: list[int | float]):
        """Initialise class"""
        if len(set(map(len, args))) != 1:
            raise ValueError("All input arrays must have the same length")
        arr = np.array(args)
        if arr.ndim != 2 or arr.shape[0] != 2:
            raise ValueError("args of XY should be two 1d arrays")
        return np.asarray(arr).view(cls)

    @property
    def x(self) -> Self:
        """Property for x array of plotting series"""
        return self[0].flatten()

    @property
    def y(self) -> Self:
        """Property for y array of plotting series"""
        return self[1].flatten()

    @y.setter
    def y(self, val: np.ndarray):
        """Setter for y property"""
        if isinstance(val, (np.ndarray, list)):
            if len(val) != len(self.y):
                raise ValueError("Cannot modify shape of XY.y")
        else:
            raise TypeError("Expected input to be either a list or numpy.ndarray")
        self[1] = val

    def find_highest_peak(self) -> tuple[int, int | float]:
        """Finds the index and x-value of the highest peak in the y-array.

        Returns:
            tuple[int, int | float]: Tuple containing the index of the highest peak and its corresponding x-value.
        """
        peaks, props = find_peaks(self.y, height=0)
        idx_peak = peaks[np.argmax(props["peak_heights"])]
        x_peak = self.x[idx_peak]
        return idx_peak, x_peak

    def get_union_x_range(self, other_xy) -> np.ndarray:
        """Gets Union x-range from this XY object and another.

        Args:
            other_xy (XY): Other XY object to compare with.

        Returns:
            np.ndarray: np.arange of Union x-range of self and other_xy.
        """
        x_min = min(self.x.min(), other_xy.x.min())
        x_max = max(self.x.max(), other_xy.x.max())
        x_range = np.arange(x_min, x_max + 1)
        return x_range
