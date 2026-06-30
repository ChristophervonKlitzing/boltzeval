import numpy as np

from boltzeval.utils.histogram import Histogram


def get_energy_hist(
    log_probs: np.ndarray,
    n_bins: int = 100,
    energy_range: tuple[float, float] | None = None,
    quantile_range: tuple[float, float] | None = (0.01, 0.99),
    margin_ratio: float = 0.5,
):
    """
    Compute a histogram of reduced energies from log-probabilities.

    The reduced energy (dimensionless) is defined as

        u(x) = -log p(x),

    i.e., the negative log-probability, where p can be unnormalized.

    For a Boltzmann distribution

        p(x) ∝ exp(-E(x) / (k_B T)),

    the reduced energy corresponds to

        u(x) = E(x) / (k_B T).

    Only relative differences in reduced energy are physically meaningful,
    since the normalization constant Z = ∫ p(x) dx is generally unknown.

    Parameters
    ----------
    log_probs : np.ndarray
        Array of log-probabilities log p(x). Can have arbitrary shape;
        the histogram is computed over all elements.

    n_bins : int, default=100
        Number of bins for the histogram.

    energy_range : tuple[float, float] | None, default=None
        Explicit (min, max) range for the histogram. If provided,
        `quantile_range` is ignored.

    quantile_range : tuple[float, float] | None, default=(0.01, 0.99)
        Lower and upper quantiles used to automatically determine the
        histogram range if `energy_range` is None. This is useful to
        exclude extreme outliers. If None, the full data range is used.

    margin_ratio : float, default=0.5
        Fraction of the data range to add as padding on both sides of the
        histogram range when it is determined from `quantile_range`. For
        example, a value of 0.5 extends the lower and upper bounds by
        50% of the selected quantile width, providing extra margin to
        ensure edge data points are included, which are not extreme outliers.

    Returns
    -------
        Histogram1D
        A Histogram1D object containing the histogram and bin edges
    """

    reduced_energy = -log_probs.flatten()

    # Determine histogram range
    if energy_range is None:
        if quantile_range is not None:
            energy_range = determine_energy_hist_range(
                log_probs, quantile_range=quantile_range, margin_ratio=margin_ratio
            )
        else:
            energy_range = (reduced_energy.min(), reduced_energy.max())

    return Histogram.from_samples(
        reduced_energy,
        bins=n_bins,
        data_range=energy_range,
    )


def determine_energy_hist_range(
    log_probs: np.ndarray,
    quantile_range: tuple[float, float] = (0.01, 0.99),
    margin_ratio: float = 0.5,
):
    """
    Determine a histogram energy range from reduced-energy quantiles.

    Parameters
    ----------
    log_probs : np.ndarray
        Array of log-probabilities log p(x).

    quantile_range : tuple[float, float], default=(0.01, 0.99)
        Lower and upper quantiles used to define the initial range.

    margin_ratio : float, default=0.5
        Fraction of the quantile width added as padding to both ends.

    Returns
    -------
    tuple[float, float]
        Recommended histogram range for reduced energies.
    """

    reduced_energy = -log_probs.flatten()

    q_low, q_high = quantile_range
    energy_range_mutable = [
        np.quantile(reduced_energy, q_low),
        np.quantile(reduced_energy, q_high),
    ]
    width = energy_range_mutable[1] - energy_range_mutable[0]
    energy_range_mutable[0] = energy_range_mutable[0] - width * margin_ratio
    energy_range_mutable[1] = energy_range_mutable[1] + width * margin_ratio

    energy_range = tuple(energy_range_mutable)
    return energy_range
