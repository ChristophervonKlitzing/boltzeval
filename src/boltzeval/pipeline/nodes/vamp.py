from boltzeval.metrics.feature_transforms import FeatureTransform
from boltzeval.pipeline import EvaluationNode
from boltzeval.metrics.vamp import get_vamp_r_score


class VampNode(EvaluationNode):
    requirements = ["trajs_true", "trajs_pred"]

    def __init__(
        self,
        lag: int,
        feature_transform: FeatureTransform,
        r=2.0,
    ):
        super().__init__()

        self._lag = lag
        self._feature_transform = feature_transform
        self._r = r

    def _eval(self, data):
        trajs_true = data.trajs_true
        trajs_pred = data.trajs_pred

        vamp_r_true = get_vamp_r_score(
            trajs_true,
            lag=self._lag,
            feature_transform=self._feature_transform,
            r=self._r,
        )
        vamp_r_pred = get_vamp_r_score(
            trajs_pred,
            lag=self._lag,
            feature_transform=self._feature_transform,
            r=self._r,
        )

        metrics = {
            f"vamp/vamp_{self._r}_lag_{self._lag}_gap": vamp_r_true - vamp_r_pred,
            f"vamp/vamp_{self._r}_lag_{self._lag}_true": vamp_r_true,
            f"vamp/vamp_{self._r}_lag_{self._lag}_pred": vamp_r_pred,
        }

        return metrics
