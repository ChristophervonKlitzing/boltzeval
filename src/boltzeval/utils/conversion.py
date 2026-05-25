import numpy as np


def to_free_energy(hist: np.ndarray, shift_min: bool = False) -> np.ndarray:
    """
    Convert histogram counts into free energy values (in units of kT).
    Normalization of hist is not important as this function normalizes them.

    Parameters
    ----------
    hist : np.ndarray
        Histogram counts.
    shift_min : bool, optional
        If True, shifts the minimum free energy to zero.

    Returns
    -------
    np.ndarray
        Free energy values corresponding to histogram counts.
    """

    # Free energy is based on probability of being in bin[i,j] not on its density
    # -> the bin area should therefore not be included in the normalization.
    probs = hist / hist.sum()

    # Prevent divide-by-zero warnings from np.log by replacing zeros with a tiny positive value.
    # Zero entries correspond to infinite free energy, so the exact replacement value does not matter.
    mask = probs > 0
    probs[~mask] = 1e-300

    # Compute free energy for nonzero probabilities
    fe = np.where(mask, -np.log(probs), np.inf)

    if shift_min:
        fe -= np.min(fe[np.isfinite(fe)])

    return fe
