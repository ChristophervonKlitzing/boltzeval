import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker

from boltzeval.metrics.feature_transforms import FeatureTransform
from boltzeval.pipeline import EvaluationNode
from boltzeval.metrics.vamp import get_implied_timescales, get_vamp_r_score_pair
from boltzeval.utils.pdf import PdfBuffer, matplotlib_to_pdf_buffer
from boltzeval.utils.trajectory import TrajectoryEnsemble


def _lag_label(lag_time: float, trajectories: TrajectoryEnsemble) -> str:
    """Metric-key friendly label of a lag time, e.g. '0.5ps' or '0.5'."""
    unit = trajectories.time_unit
    return f"{lag_time:g}" + (unit if unit is not None else "")


class VampNode(EvaluationNode):
    """
    Compares the VAMP-r score of predicted trajectories against reference ones.

    The lag is specified as a physical lag time, not as a number of frames:
    VAMP scores are only comparable when evaluated at the same physical lag,
    while reference and predicted trajectories may well be saved at different
    frame strides. The requested lag time must be an integer multiple of the
    frame stride of *both* ensembles.
    """

    requirements = ["trajs_true", "trajs_pred"]

    def __init__(
        self,
        lag_time: float,
        feature_transform: FeatureTransform,
        r=2.0,
    ):
        super().__init__()

        self._lag_time = lag_time
        self._feature_transform = feature_transform
        self._r = r

    def _eval(self, data):
        trajs_true = data.trajs_true
        trajs_pred = data.trajs_pred

        vamp_r_true, vamp_r_pred = get_vamp_r_score_pair(
            trajs_true=trajs_true,
            trajs_pred=trajs_pred,
            lag_time=self._lag_time,
            feature_transform=self._feature_transform,
            r=self._r,
        )

        lag = _lag_label(self._lag_time, trajs_true)
        key = f"vamp/vamp_{self._r}_lag_{lag}"

        metrics = {
            f"{key}_gap": vamp_r_true - vamp_r_pred,
            f"{key}_true": vamp_r_true,
            f"{key}_pred": vamp_r_pred,
        }

        return metrics


def _visualize_timescales(
    timescales_true: np.ndarray,
    timescales_pred: np.ndarray,
    time_unit: str | None,
    log_scale: bool = True,
    annotate_timescales: bool = False,
) -> PdfBuffer:
    """
    Parity plot of the implied timescales: reference on x, predicted on y.

    A model that reproduces the dynamics perfectly puts every process onto the
    diagonal, which is drawn as a grey dashed line.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    all_timescales = np.concatenate([timescales_true, timescales_pred])
    # A log scale is the natural one for timescales, but it cannot show the
    # non-positive values a badly conditioned estimate may produce.
    use_log = log_scale and bool(np.all(all_timescales > 0.0))

    low = float(all_timescales.min())
    high = float(all_timescales.max())

    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
        low, high = low / 2.0, high * 2.0

        # Over a range of a few decades the default log formatter labels the
        # minor ticks as well, which overlap into an unreadable smear. Label
        # the 1/2/5 decades instead, so a narrow range still gets a few ticks.
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            axis.set_major_formatter(ticker.ScalarFormatter())
            axis.set_minor_formatter(ticker.NullFormatter())
    else:
        padding = 0.1 * (high - low) if high > low else 0.1 * max(abs(high), 1.0)
        low, high = low - padding, high + padding

    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
        color="grey",
        linewidth=1.0,
        zorder=0,
        label="ideal",
    )

    ax.scatter(
        timescales_true,
        timescales_pred,
        s=40,
        color="crimson",
        marker="o",
        zorder=1,
    )

    # Label every point with the index of its process (0 = slowest)
    if annotate_timescales:
        for i, (its_true, its_pred) in enumerate(zip(timescales_true, timescales_pred)):
            ax.annotate(
                str(i),
                (its_true, its_pred),
                textcoords="offset points",
                xytext=(7, 4),
                fontsize=9,
            )

    unit = f" / {time_unit}" if time_unit is not None else ""
    ax.set_xlabel(f"reference timescale{unit}")
    ax.set_ylabel(f"predicted timescale{unit}")

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal")
    ax.legend()

    pdf = matplotlib_to_pdf_buffer(fig)
    plt.close(fig)
    return pdf


class ImpliedTimescaleNode(EvaluationNode):
    """
    Compares the slowest implied (relaxation) timescales of predicted
    trajectories against reference ones.

    Timescales are reported in physical time units, which requires the frame
    stride of each ensemble: the same number of frames means a different
    relaxation time for trajectories saved at different strides.

    Optionally also produces a parity plot of the timescales (reference on the
    x-axis, predicted on the y-axis), in which a perfect model lies on the
    diagonal.

    By default all available timescales are compared, which is one per feature
    dimension; ``n_timescales`` restricts the comparison to the slowest ones.
    """

    requirements = ["trajs_true", "trajs_pred"]

    def __init__(
        self,
        lag_time: float,
        feature_transform: FeatureTransform,
        n_timescales: int | None = None,
        include_pdf: bool = True,
        log_scale: bool = True,
        annotate_timescales: bool = False,
    ):
        super().__init__()

        self._lag_time = lag_time
        self._feature_transform = feature_transform
        self._n_timescales = n_timescales
        self._include_pdf = include_pdf
        self._log_scale = log_scale
        self._annotate_timescales = annotate_timescales

    def _eval(self, data):
        trajs_true = data.trajs_true
        trajs_pred = data.trajs_pred

        timescales_true = get_implied_timescales(
            trajs_true,
            lag_time=self._lag_time,
            feature_transform=self._feature_transform,
            n_timescales=self._n_timescales,
            name="trajs_true",
        )
        timescales_pred = get_implied_timescales(
            trajs_pred,
            lag_time=self._lag_time,
            feature_transform=self._feature_transform,
            n_timescales=self._n_timescales,
            name="trajs_pred",
        )

        # A rank-deficient estimate yields fewer timescales than the feature
        # dimension, and it may do so for only one of the two ensembles.
        n_common = min(len(timescales_true), len(timescales_pred))
        if len(timescales_true) != len(timescales_pred):
            warnings.warn(
                f"'{type(self).__name__}' obtained {len(timescales_true)} timescales "
                f"for trajs_true but {len(timescales_pred)} for trajs_pred; comparing "
                f"the {n_common} slowest ones.",
                UserWarning,
                stacklevel=2,
            )
            timescales_true = timescales_true[:n_common]
            timescales_pred = timescales_pred[:n_common]

        lag = _lag_label(self._lag_time, trajs_true)
        group = f"timescales/its_lag_{lag}"

        metrics = {}
        for i, (its_true, its_pred) in enumerate(zip(timescales_true, timescales_pred)):
            metrics[f"{group}_{i}_true"] = float(its_true)
            metrics[f"{group}_{i}_pred"] = float(its_pred)
            # Relative deviation: timescales easily span orders of magnitude,
            # so an absolute difference is hard to read across systems.
            metrics[f"{group}_{i}_rel_error"] = float(
                abs(its_pred - its_true) / abs(its_true)
            )

        if self._include_pdf:
            metrics[f"{group}_pdf"] = _visualize_timescales(
                timescales_true=timescales_true,
                timescales_pred=timescales_pred,
                time_unit=trajs_true.time_unit,
                log_scale=self._log_scale,
                annotate_timescales=self._annotate_timescales,
            )

        return metrics
