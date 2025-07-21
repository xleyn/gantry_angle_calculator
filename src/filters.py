import numpy as np


def bilateral_filter_1d(signal, sigma_color: float, sigma_space: float):
    filtered_signal = []
    window_radius = int(3 * sigma_space)
    signal_padded = np.pad(signal, window_radius, mode="edge")
    kernel_spatial = np.exp(
        -((np.arange(-window_radius, window_radius + 1)) ** 2) / (2 * sigma_space**2)
    )

    for i in range(len(signal)):
        window = signal_padded[i : i + 2 * window_radius + 1]
        signal_centre = signal[i]

        numerator = 0
        denominator = 0

        for j in range(len(window)):
            weight_spatial = kernel_spatial[j]
            weight_color = np.exp(
                -((window[j] - signal_centre) ** 2) / (2 * sigma_color**2)
            )
            weight = weight_spatial * weight_color
            numerator += weight * window[j]
            denominator += weight

        filtered_signal.append(
            numerator / denominator if denominator != 0 else signal_centre
        )

    return np.array(filtered_signal)
