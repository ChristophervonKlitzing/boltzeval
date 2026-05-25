from collections import defaultdict
from typing import Protocol, TypeVar

import numpy as np

from boltzeval.utils.histogram import Histogram


class HistogramMetric(Protocol):
    """
    Protocol for histogram distance or divergence metrics.

    A histogram metric is a callable object that compares two histograms
    with identical binning and returns a scalar score.

    Implementations must also define an ``id`` property used for logging
    and aggregation.
    """

    def __call__(self, hist_p: Histogram, hist_q: Histogram) -> float:
        pass

    @property
    def id(self) -> str:
        pass


def get_hist_fwd_kullback_leibler(hist_p: Histogram, hist_q: Histogram):
    """
    Compute the forward Kullback-Leibler divergence between two histograms.

    The divergence is defined as:

    .. math::

        KL(P \\parallel Q) = \\int P(x) \\log \\frac{P(x)}{Q(x)} dx

    and approximated using a discrete sum over histogram bins.

    A small constant is added for numerical stability.

    Parameters
    ----------
    hist_p : Histogram1D or Histogram2D
        Reference distribution :math:`P`.
    hist_q : Histogram1D or Histogram2D
        Approximation distribution :math:`Q`. Must share identical binning.

    Returns
    -------
    float
        Estimated KL divergence :math:`KL(P \\parallel Q)`.

    Notes
    -----
    - Histograms must have identical bin structure.
    - The bin area is used to approximate the continuous integral.
    """
    assert hist_p.get_num_bins() == hist_q.get_num_bins()

    hist_density_p = hist_p.get_as_density()
    hist_density_q = hist_q.get_as_density()

    eps_ram = 1e-10
    bin_area = hist_p.get_bin_volume()

    log_ratio = np.log(hist_density_p + eps_ram) - np.log(hist_density_q + eps_ram)
    kld_ram = (
        np.sum(hist_density_p * log_ratio)
        * bin_area  # To get the properly normalized integral / KLD
    )
    return float(kld_ram)


get_hist_fwd_kullback_leibler.id = "kl"


def get_hist_total_variation(hist_p: Histogram, hist_q: Histogram):
    """
    Compute the total variation distance between two histograms.

    Defined as:

    .. math::

        TV(P, Q) = \\frac{1}{2} \\int |P(x) - Q(x)| dx

    Approximated via summation over histogram bins.

    Parameters
    ----------
    hist_p : Histogram1D or Histogram2D
        First distribution :math:`P`.
    hist_q : Histogram1D or Histogram2D
        Second distribution :math:`Q`. Must share identical binning.

    Returns
    -------
    float
        Total variation distance in :math:`[0, 1]`.

    Notes
    -----
    - Requires identical binning.
    - Uses bin area for continuous approximation.
    """
    assert hist_p.get_num_bins() == hist_q.get_num_bins()

    hist_density_p = hist_p.get_as_density()
    hist_density_q = hist_q.get_as_density()

    bin_area = hist_p.get_bin_volume()

    total_variation_ram = (
        0.5 * np.sum(np.abs(hist_density_p - hist_density_q)) * bin_area
    )  # To get the properly normalized integral / KLD
    return float(total_variation_ram)


get_hist_total_variation.id = "tv"


def get_hist_jensen_shannon(hist_p: Histogram, hist_q: Histogram):
    """
    Compute the Jensen-Shannon divergence between two histograms.

    Defined as:

    .. math::

        JSD(P, Q) = \\frac{1}{2} KL(P \\parallel M) + \\frac{1}{2} KL(Q \\parallel M)

    where:

    .. math::

        M = \\frac{1}{2}(P + Q)

    Parameters
    ----------
    hist_p : Histogram1D or Histogram2D
        First distribution :math:`P`.
    hist_q : Histogram1D or Histogram2D
        Second distribution :math:`Q`. Must share identical binning.

    Returns
    -------
    float
        Jensen-Shannon divergence (non-negative, symmetric).

    Notes
    -----
    - Histograms must have identical binning.
    - A small epsilon is used for numerical stability.
    """
    assert hist_p.get_num_bins() == hist_q.get_num_bins()

    hist_density_p = hist_p.get_as_density()
    hist_density_q = hist_q.get_as_density()

    eps_ram = 1e-10
    bin_area = hist_p.get_bin_volume()

    m = 0.5 * (hist_density_p + hist_density_q)

    kl_pm = np.sum(
        hist_density_p * (np.log(hist_density_p + eps_ram) - np.log(m + eps_ram))
    )

    kl_qm = np.sum(
        hist_density_q * (np.log(hist_density_q + eps_ram) - np.log(m + eps_ram))
    )

    jsd_ram = 0.5 * (kl_pm + kl_qm) * bin_area

    return float(jsd_ram)


get_hist_jensen_shannon.id = "jsd"


def eval_histogram_metrics(
    hist_metrics: list[HistogramMetric],
    true: list[Histogram],
    pred: list[Histogram],
    *,
    group: str,
    hist_type: str | None = None,
    include_individual: bool = True,
    include_aggregated: bool = True,
):
    """
    Evaluate multiple histogram metrics on pairs of histograms.

    Each metric is applied to corresponding pairs from ``true`` and ``pred``.
    Results are returned as a flat dictionary with structured keys.

    Keys follow the format:

    .. code-block:: text

        {group}/{h_type}_{metric_id}_{i}
        {group}/{metric_id}_{i}          (if h_type is None)

    where `i` is the index of the histogram pair in `true` and `pred`.

    Aggregated statistics (mean over histogram pairs) are optionally included:

    .. code-block:: text

        {group}/{hist_type}_{metric_id}_mean
        {group}/{metric_id}_mean

    Parameters
    ----------
    hist_metrics : list[HistogramMetric]
        Metrics to evaluate.
    true : list[Histogram1D or Histogram2D]
        Ground-truth histograms.
    pred : list[Histogram1D or Histogram2D]
        Predicted histograms (same length as ``true``).
    group : str
        Metric group name (e.g., ``"torsion"``).
    hist_type : str, optional
        Histogram type identifier (e.g., ``"phi_psi"``).
    include_individual : bool, default=True
        Whether to include per-histogram metric values.
    include_aggregated : bool, default=True
        Whether to include mean values across histogram pairs.

    Returns
    -------
    dict[str, float]
        Dictionary of computed metrics.

    Notes
    -----
    - ``true`` and ``pred`` must have equal length.
    - Metrics are evaluated pairwise using ``zip``.
    """

    assert include_aggregated or include_individual

    metrics = {}
    summed_metrics: dict[str, list[float]] = defaultdict(list)
    _type = f"{hist_type}_" if hist_type is not None else ""

    # Iterate over the torsion backbone angle pairs
    for i, (true_hist, pred_hist) in enumerate(zip(true, pred)):
        # evaluate all specified metrics on the angle histogram pairs
        for metric in hist_metrics:
            m_i = metric(true_hist, pred_hist)

            if include_individual:
                metrics[f"{group}/{_type}{metric.id}_{i}"] = m_i
            summed_metrics[metric.id].append(m_i)

    if include_aggregated:
        for id, metric_values in summed_metrics.items():
            metrics[f"{group}/{_type}{id}_mean"] = np.array(metric_values).mean()

    return metrics
