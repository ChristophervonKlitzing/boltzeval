import numpy as np
import deeptime as dt

from boltzeval.metrics.feature_transforms import FeatureTransform


def get_vamp2_score(
    trajectories: list[np.ndarray], lag: int, feature_transform: FeatureTransform
):
    featurized_trajs = np.stack([feature_transform(t) for t in trajectories])
    # shape (#trajs, traj_length, #atoms, 3)
    vamp = dt.decomposition.VAMP(lag).fit(featurized_trajs).fetch_model()
    vamp2_score = vamp.score(2)
    return vamp2_score
