import numpy as np


def squeeze_last_dim(x: np.ndarray) -> np.ndarray:
    """
    Squeeze the last dimension of an array if it has size 1.

    Converts arrays of shape (batch, 1) to (batch,), leaves (batch,) unchanged.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (batch,) or (batch, 1).

    Returns
    -------
    np.ndarray
        Flattened array of shape (batch,).
    """
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        return np.squeeze(x, -1)
    elif x.ndim == 1:
        return x
    else:
        raise ValueError(f"Input must have shape (batch,) or (batch, 1), got {x.shape}")
