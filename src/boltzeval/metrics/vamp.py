import numpy as np
import deeptime as dt

from boltzeval.metrics.feature_transforms import FeatureTransform


def get_vamp_r_score(
    trajectories: list[np.ndarray] | np.ndarray,
    lag: int,
    feature_transform: FeatureTransform,
    r: float = 2.0,
):
    featurized_trajs = np.stack([feature_transform(t) for t in trajectories])
    # shape (#trajs, traj_length, #atoms, 3)
    vamp = dt.decomposition.VAMP(lag).fit(featurized_trajs).fetch_model()
    vamp_r_score = vamp.score(r)
    return vamp_r_score
