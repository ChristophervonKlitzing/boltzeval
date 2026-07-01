import numpy as np
from scipy.special import logsumexp as _logsumexp

from boltzeval.utils.shape_utils import squeeze_last_dim


def get_reverse_ess(log_weights: np.ndarray) -> float:
    """
    Compute the reverse effective sample size (ESS) from log-importance weights.

    Reverse ESS measures how well a proposal distribution :math:`q(x)` covers a
    target distribution :math:`p(x)` in importance sampling.

    Using weights :math:`w_i = \\exp(\\log p(x_i) - \\log q(x_i))`, the ESS is:

    .. math::

        \\mathrm{ESS} = \\frac{\\left(\\sum_i w_i\\right)^2}{N \\sum_i w_i^2}

    Log-form computation is used for numerical stability.

    Parameters
    ----------
    log_weights : np.ndarray
        Log importance weights:

        .. math::

            \\log w_i = \\log p(x_i) - \\log q(x_i)

        where :math:`x_i \\sim q(x)`.

    Returns
    -------
    float
        Reverse ESS in :math:`(0, 1]`, representing the effective fraction of
        useful samples.
    """
    log_weights = squeeze_last_dim(log_weights)

    logN = np.log(log_weights.shape[0])

    log_sum_w = _logsumexp(log_weights, axis=0)
    log_sum_w2 = _logsumexp(2.0 * log_weights, axis=0)

    log_ess = 2.0 * log_sum_w - logN - log_sum_w2
    return float(np.exp(log_ess))


def get_forward_ess(log_weights: np.ndarray) -> float:
    """
    Compute the forward effective sample size (ESS) from log-importance weights.

    Forward ESS measures overlap quality when samples are drawn from the target
    distribution :math:`p(x)` and reweighted toward a proposal :math:`q(x)`.

    It is defined as:

    .. math::

        \\mathrm{ESS} =
        \\frac{1}{\\mathbb{E}_p[w] \\; \\mathbb{E}_p[1/w]}

    where :math:`w = \\exp(\\log p(x) - \\log q(x))`.

    Parameters
    ----------
    log_weights : np.ndarray
        Log importance weights:

        .. math::

            \\log w_i = \\log p(x_i) - \\log q(x_i)

        where :math:`x_i \\sim p(x)`.

    Returns
    -------
    float
        Forward ESS in :math:`(0, 1]`, indicating how well the proposal
        distribution represents the target distribution.
    """
    log_weights = squeeze_last_dim(log_weights)

    logN = np.log(log_weights.shape[0])
    log_z_inv = _logsumexp(-log_weights, 0) - logN
    log_z_expectation_p_over_q = _logsumexp(log_weights, 0) - logN
    log_forward_ess = -(log_z_inv + log_z_expectation_p_over_q)
    return float(np.exp(log_forward_ess))


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Number of samples
    N = 1_000

    # ------------------------------------------------------------------
    # Samples x ~ q(x): model distribution (normalized)
    # ------------------------------------------------------------------
    # q = N(0, 1)
    model_samples = rng.normal(loc=0.0, scale=1.0, size=N)

    # log q(x) -- normalized density
    model_log_prob = -0.5 * model_samples**2 - 0.5 * np.log(2.0 * np.pi)

    # ------------------------------------------------------------------
    # Target distribution p(x): unnormalized
    # ------------------------------------------------------------------
    # p(x) ∝ exp(-0.5 * (x - 2)^2 / 0.5^2)
    # (intentionally NOT normalized)
    target_log_prob = -0.5 * ((model_samples - 1.0) / 0.5) ** 2

    # ------------------------------------------------------------------
    # Importance log-weights
    # ------------------------------------------------------------------
    log_weights = target_log_prob - model_log_prob

    # ------------------------------------------------------------------
    # Compute ESS
    # ------------------------------------------------------------------
    reverse_ess = get_reverse_ess(log_weights)

    target_samples = rng.normal(loc=1.0, scale=0.5, size=N)
    target_log_prob_fwd = -0.5 * ((target_samples - 1.0) / 0.5) ** 2
    model_log_prob_fwd = -0.5 * target_samples**2 - 0.5 * np.log(2.0 * np.pi)
    fwd_log_weights = target_log_prob_fwd - model_log_prob_fwd
    forward_ess = get_forward_ess(fwd_log_weights)

    print("Gaussian importance sampling example")
    print("-----------------------------------")
    print(f"Reverse ESS           : {reverse_ess*100:.6f}%")
    print(f"Forward ESS           : {forward_ess*100:.6f}%")
