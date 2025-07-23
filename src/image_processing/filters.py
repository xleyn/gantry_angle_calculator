"""
filters.py

This script contains various importable custom filters to be used in
1D signal processing/convolution.

Written by Nathan Crossley, 2025.
"""

import numpy as np


def bilateral_filter_1d(
    signal: np.ndarray, sigma_color: float, sigma_space: float
) -> np.ndarray:
    """Applies a bilateral filter to a 1D array.
    This is a convolution, taking the weighted average of neighbouring pixels,
    but with weights defined by both spatial and intensity based kernels.
    Smoothing is reduced where signal intensity changes rapidly, thus making
    this filter ideal for edge preservation.

    Args:
        signal (np.ndarray): Signal to apply bilateral filter to.
        sigma_color (float): Standard deviation for color/intensity smoothing kernel
        sigma_space (float): Standard deviation for spatial smoothign kernel

    Returns:
        np.ndarray: Filtered array.
    """

    # initialise list to hold filtered signal
    filtered_signal = []

    # define window radius for pixel neighbourhood
    window_radius = int(3 * sigma_space)

    # pad signal by window radius to ensure that smoothing is effective for edges of signal
    signal_padded = np.pad(signal, window_radius, mode="edge")

    # calculate spatial kernel beforehand (dependent on position only)
    kernel_spatial = np.exp(
        -((np.arange(-window_radius, window_radius + 1)) ** 2) / (2 * sigma_space**2)
    )

    # iterate over each point in signal
    for i in range(len(signal)):

        # define pixel neighbourhood from padded signal and get signal at index from original signal
        window = signal_padded[i : i + 2 * window_radius + 1]
        signal_centre = signal[i]

        # Initialise numerator and denominator for later calcs
        numerator = 0
        denominator = 0

        # iterate over pixels in neighbourhood
        for j in range(len(window)):

            # calculate spatial and color weights according to bilateral filtering theory
            weight_spatial = kernel_spatial[j]
            weight_color = np.exp(
                -((window[j] - signal_centre) ** 2) / (2 * sigma_color**2)
            )

            # calculate combined weight by product of weights
            weight = weight_spatial * weight_color

            # add contribution of pixel in neighbourhood to numerator and denominator
            numerator += weight * window[j]
            denominator += weight

        # calculate filtered signal magnitude by numerator / denominator
        filtered_signal.append(
            numerator / denominator if denominator != 0 else signal_centre
        )

    return np.array(filtered_signal)
