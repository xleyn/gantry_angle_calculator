import warnings
import numpy as np
from connected_peaks_model.xy import XY
from scipy.optimize import curve_fit


class Sigmoid(XY):
    """Class for fitting a sigmoid model to an XY object."""

    def __init__(self, *args: list[int | float]):
        """Initialises Sigmoid class. Positional argumments should be x and y arrays."""
        self.popt = self.get_popt()

    def get_popt(self) -> list[float]:
        """Gets best fit parameters for sigmoid fitting.

        Returns:
            list[float]: List of best fit parameters for sigmoid fitting.
        """
        p0 = self.get_initial_guesses()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try:
                popt, _ = curve_fit(self.functional_form, self.x, self.y, p0=p0)
            except RuntimeError:
                print(
                    f"User warning: Runtime error in sigmoid fitting. Using initial guesses."
                )
                popt = p0
        return popt

    def get_initial_guesses(self) -> list[float]:
        """Gets initial guesses for sigmoid fitting.

        Returns:
            list[float]: List of initial guesses for sigmoid fitting.
        """
        A = np.ptp(self.y)
        b = np.min(self.y)
        abs_deriv = np.abs(np.diff(self.y) / np.diff(self.x))
        abs_grad_max = np.max(abs_deriv)
        k = np.sign(self.y[-1] - self.y[0]) * abs_grad_max * 0.5
        x0 = self.x[np.argmax(abs_deriv)]
        p0 = [A, k, x0, b]
        return p0

    @staticmethod
    def functional_form(
        x: list[float] | float, A: float, k: float, x0: float, b: float
    ) -> list[float] | float:
        """Evaluates the functional form of the sigmoid for either an input x-array or a single x value.

        Args:
            x (list[float] | float): Input x-array or single x value.
            A (float): Amplitude parameter for sigmoid model.
            k (float): Steepness parameter for sigmoid model.
            x0 (float): X-centre parameter for sigmoid model.
            b (float): y-offset parameter for sigmoid model.

        Returns:
            list[float] | float: y-array or single y value for input x.
        """
        exp_term = np.exp(-k * (x - x0))
        return A / (1 + exp_term) + b

    def evaluate(self, x: list) -> XY:
        """Evaluates the sigmoid model for an input x-array.

        Args:
            x (list): Input x-array.

        Returns:
            XY: Output XY object for input x-array using fitted sigmoid model.
        """
        return XY(x, self.functional_form(x, *self.popt))
