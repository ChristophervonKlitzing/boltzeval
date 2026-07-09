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


def reshape_to_molecular(obj: np.ndarray):
    """
    Savely tries to reshape obj into shape (batch, n_atoms, 3). Also handles
    batch sizes of zero and throws an error if something doesn't work out.
    """
    if obj.ndim == 2:
        # Be explicit (not use -1) to also handle batch_size = 0
        batch_size, dim = obj.shape
        assert dim % 3 == 0
        return obj.reshape(batch_size, dim // 3, 3)
    elif obj.ndim != 3:
        raise ValueError(f"Invalid input shape {obj.shape}")
    return obj
