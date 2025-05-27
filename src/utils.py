from scipy import signal
import numpy as np


def find_highest_peak(y: list[float]) -> int:
    """Finds highest x-idx of highest peak in input y-array.

    Args:
        y (list[float]): y-array for peak detection.

    Returns:
        int: idx of highest peak in input y-array.
    """
