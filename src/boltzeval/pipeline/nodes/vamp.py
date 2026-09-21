from boltzeval.metrics.feature_transforms import FeatureTransform
from boltzeval.pipeline import EvaluationNode
from boltzeval.metrics.vamp import get_implied_timescales, get_vamp_r_score_pair
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


class ImpliedTimescaleNode(EvaluationNode):
    """
    Compares the slowest implied (relaxation) timescales of predicted
    trajectories against reference ones.

    Timescales are reported in physical time units, which requires the frame
    stride of each ensemble: the same number of frames means a different
    relaxation time for trajectories saved at different strides.
    """

    requirements = ["trajs_true", "trajs_pred"]

    def __init__(
        self,
        lag_time: float,
        feature_transform: FeatureTransform,
        n_timescales: int = 2,
    ):
        super().__init__()

        self._lag_time = lag_time
        self._feature_transform = feature_transform
        self._n_timescales = n_timescales

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

        return metrics
