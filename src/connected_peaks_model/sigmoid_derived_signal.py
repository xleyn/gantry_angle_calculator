import numpy as np

from connected_peaks_model.sigmoid import Sigmoid
from connected_peaks_model.xy import XY


class SigmoidDerivedSignal(XY):
    """Subclass of XY that represents a profile derived from two or more sigmoids.
    As such, the left and right sides of the profile are each functionally defined by
    a sigmoid, which is useful for extrapolation and blending."""

    def __new__(cls, sig_L: Sigmoid, sig_R: Sigmoid, signal: XY | None = None):
        """Creates a new instance of SigmoidDerivedSignal with two sigmoids functionally defining the
        left and right sides of the profile. If a signal is not provided, it is derived
        from the sigmoids.

        Args:
            sig_L (Sigmoid): Sigmoid defining the far left side of the signal.
            sig_R (Sigmoid): Sigmoid defining the far right side of the signal.
            signal (XY | None, optional): Describes the shape of the profile. Defaults to None.
        """
        if signal is None:
            signal = cls.setup_from_sigmoids(sig_L, sig_R)

        obj = super().__new__(cls, signal.x, signal.y)
        return obj

    def __init__(self, sig_L: Sigmoid, sig_R: Sigmoid, signal: XY | None = None):
        """Initialises SigmoidDerivedSignal with two sigmoids functionally defining the
        left and right sides of the profile. If a signal is not provided, it is derived
        from the sigmoids.

        Args:
            sig_L (Sigmoid): Sigmoid defining the far left side of the signal.
            sig_R (Sigmoid): Sigmoid defining the far right side of the signal.
            signal (XY | None, optional): Describes the shape of the profile. Defaults to None.
        """
        if signal is None:
            signal = self.setup_from_sigmoids(sig_L, sig_R)
        self.sig_L = sig_L
        self.sig_R = sig_R

    @staticmethod
    def setup_from_sigmoids(sig_L: Sigmoid, sig_R: Sigmoid) -> XY:
        """Creates a new XY object based on blending two sigmoids together.

        Args:
            sig_L (Sigmoid): Sigmoid defining the far left side of the signal.
            sig_R (Sigmoid): Sigmoid defining the far right side of the signal.

        Returns:
            XY: Resultant raw XY profile.
        """
        x_range = sig_L.get_union_x_range(sig_R)

        sig_L_extrap = sig_L.evaluate(x_range)
        sig_R_extrap = sig_R.evaluate(x_range)

        transition_x = (sig_L.x[-1] + sig_R.x[0]) / 2
        transition_width = (
            1
            / 10
            * (abs(transition_x - sig_L.popt[2]) + abs(transition_x - sig_R.popt[2]))
        )

        blending_weight = 1 / (
            1 + np.exp([-(x - transition_x) / transition_width for x in x_range])
        )
        xy_blended = XY(
            x_range,
            (1 - blending_weight) * sig_L_extrap.y + blending_weight * sig_R_extrap.y,
        )

        return xy_blended

    def extrapolate(self, extrap_range: list[float]) -> XY:
        """Extrapolates self within extrap_range using the functional sigmoids defined
        at the edges of the signal.

        Args:
            extrap_range (list[float]): X-range to extrapolate self to fit.

        Returns:
            XY: Extrapolated signal within extrap_range.
        """
        defined_x_range = self.sig_L.get_union_x_range(self.sig_R)

        left_extrap = self.sig_L.evaluate(
            np.arange(extrap_range[0], defined_x_range[0])
        )
        right_extrap = self.sig_R.evaluate(
            np.arange(defined_x_range[-1] + 1, extrap_range[-1] + 1)
        )

        x_combined = np.concatenate([left_extrap.x, self.x, right_extrap.x])
        y_combined = np.concatenate([left_extrap.y, self.y, right_extrap.y])

        return XY(x_combined, y_combined)
