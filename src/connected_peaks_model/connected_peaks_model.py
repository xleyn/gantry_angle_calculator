import itertools

import numpy as np
from scipy.signal import find_peaks

from connected_peaks_model.xy import XY
from connected_peaks_model.sigmoid import Sigmoid
from connected_peaks_model.sigmoid_derived_signal import SigmoidDerivedSignal


class ConnectedPeaksModel:
    """Container class for fitting a piecewise sigmoid model to an XY object with a characteristic
    profile of connected peaks. Fitting an appropriate model is essential as the noise is
    too great for reliable use of traditional peak detection methods.
    """

    def __init__(self, signal: XY):
        """Initialises ConnectedPeaksModel

        Args:
            signal (XY): Signal to fit the model to.
        """
        self.signal = signal
        self.model = None

    def run(self) -> XY:
        """Entry point for running the connected peaks model fitting.
        Splits the signal into segments, fits sigmoids to half peaks, and blends them together
        to create a continuous model of the connected peaks. XY profile of the fitted model is
        returned for convenience.

        Returns:
            XY: Raw XY profile of the fitted model.
        """
        self.segments_peaks = self.split_into_peaks()
        self.segments_half_peaks = self.split_into_half_peaks()
        self.sigmoids = [Sigmoid(seg.x, seg.y) for seg in self.segments_half_peaks]
        self.peaks = [
            SigmoidDerivedSignal(sig1, sig2)
            for sig1, sig2 in zip(self.sigmoids[::2], self.sigmoids[1::2])
        ]
        for peak in self.peaks:
            self.blend_in_peak(peak)

        return self.model

    def split_into_peaks(self) -> list[XY]:
        """Splits the signal into segments based on detected troughs, which are used to identify
        the boundaries of the peaks. The segments are returned as a list of XY objects.

        Returns:
            list[XY]: Returned segments of the original signal, each representing a peak.
        """
        troughs, _ = troughs, _ = find_peaks(
            -self.signal.y,
            height=float(np.min(-self.signal.y)),
            prominence=float(np.ptp(self.signal.y) / 4),
            distance=len(self.signal.y) // 30,
        )
        split_indices = sorted([0, len(self.signal.y)] + troughs.tolist())
        segments_peaks = [
            self.signal[:, start:end]
            for start, end in zip(split_indices, split_indices[1:])
        ]
        return segments_peaks

    def split_into_half_peaks(self) -> list[XY]:
        """Each segment is split into two "half-peaks", in preparation for individual
        sigmoid fitting.

        Returns:
            list[XY]: List of "half-peak" segments.
        """
        split_indices = [seg.find_highest_peak()[0] for seg in self.segments_peaks]
        segments_half_peaks = [
            [seg[:, :idx], seg[:, idx:]]
            for idx, seg in zip(split_indices, self.segments_peaks)
        ]
        segments_half_peaks = list(itertools.chain(*segments_half_peaks))
        return segments_half_peaks

    def blend_in_peak(self, peak: SigmoidDerivedSignal):
        """Blends a new peak into the existing model. If the model is empty, the peak is set as the model.
        If the model already exists, the new peak is blended into the existing model by extrapolation and
        sigmoid blending about a transition point of finite width.

        Args:
            peak (SigmoidDerivedSignal): Peak to blend into the model.
        """
        if self.model is None:
            self.model = peak
        else:
            x_range = self.model.get_union_x_range(peak)
            model_extrap = self.model.extrapolate(x_range)
            peak_extrap = peak.extrapolate(x_range)
            transition_x = (self.model.x[-1] + peak.x[0]) / 2
            transition_width = (
                1
                / 10
                * (
                    abs(transition_x - self.model.sig_R.popt[2])
                    + abs(transition_x - peak.sig_L.popt[2])
                )
            )
            blending_weight = 1 / (
                1 + np.exp([-(x - transition_x) / transition_width for x in x_range])
            )
            new_model = XY(
                x_range,
                (1 - blending_weight) * model_extrap.y
                + blending_weight * peak_extrap.y,
            )
            self.model = SigmoidDerivedSignal(self.model.sig_L, peak.sig_R, new_model)
