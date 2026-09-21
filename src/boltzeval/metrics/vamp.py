import numpy as np
import deeptime as dt

from boltzeval.metrics.feature_transforms import (
    FeatureTransform,
    featurize_trajectories,
)
from boltzeval.utils.trajectory import TrajectoryEnsemble, resolve_frame_lag


def get_vamp_r_score(
    trajectories: TrajectoryEnsemble,
    lag_time: float,
    feature_transform: FeatureTransform,
    r: float = 2.0,
    name: str = "trajectories",
):
    """
    VAMP-r score of one trajectory ensemble at a physical lag time.

    The lag is given as a physical time (in the unit of the ensemble's
    ``frame_stride``) rather than in frames, so that ensembles saved at
    different strides can be scored at the same dynamical lag.

    Parameters
    ----------
    trajectories : TrajectoryEnsemble
        Trajectories to score. Their ``frame_stride`` defines how ``lag_time``
        is translated into frames.
    lag_time : float
        Lag time at which the score is evaluated. Must be an integer multiple
        of the ensemble's ``frame_stride``.
    feature_transform : FeatureTransform
        Transform applied to the frames of every trajectory before scoring.
    r : float
        Order of the VAMP score (2.0 = VAMP-2).
    name : str
        Name of the ensemble, used in error messages.

    Returns
    -------
    float
        VAMP-r score.
    """
    frame_lag = resolve_frame_lag(trajectories, lag_time, name=name)
    featurized_trajs = featurize_trajectories(trajectories, feature_transform)

    vamp = dt.decomposition.VAMP(frame_lag).fit(featurized_trajs).fetch_model()
    vamp_r_score = vamp.score(r)
    return float(vamp_r_score)


def get_vamp_r_score_pair(
    trajs_true: TrajectoryEnsemble,
    trajs_pred: TrajectoryEnsemble,
    lag_time: float,
    feature_transform: FeatureTransform,
    r: float = 2.0,
) -> tuple[float, float]:
    """
    VAMP-r scores of a reference and a predicted ensemble at the same lag time.

    VAMP scores may only be compared at an identical *physical* lag, so the
    requested ``lag_time`` is resolved against both ensembles up front: it must
    be an integer multiple of either ``frame_stride``, and fit into the
    trajectories of both ensembles. The two ensembles themselves need not share
    a frame stride.

    Returns
    -------
    tuple[float, float]
        (score of trajs_true, score of trajs_pred)
    """
    # Resolve both lags before doing any work, so an incompatible lag time
    # fails immediately rather than after the first (expensive) fit.
    resolve_frame_lag(trajs_true, lag_time, name="trajs_true")
    resolve_frame_lag(trajs_pred, lag_time, name="trajs_pred")

    vamp_r_true = get_vamp_r_score(
        trajs_true,
        lag_time=lag_time,
        feature_transform=feature_transform,
        r=r,
        name="trajs_true",
    )
    vamp_r_pred = get_vamp_r_score(
        trajs_pred,
        lag_time=lag_time,
        feature_transform=feature_transform,
        r=r,
        name="trajs_pred",
    )
    return vamp_r_true, vamp_r_pred


def get_implied_timescales(
    trajectories: TrajectoryEnsemble,
    lag_time: float,
    feature_transform: FeatureTransform,
    n_timescales: int | None = None,
    name: str = "trajectories",
) -> np.ndarray:
    """
    Implied (relaxation) timescales of a trajectory ensemble, in physical time.

    The timescales are obtained from the eigenvalues of the VAMP/Koopman model
    estimated at ``lag_time``. deeptime reports them in frames, so they are
    converted back into physical time using the ensemble's ``frame_stride``;
    this is what makes timescales of differently strided ensembles comparable.

    Parameters
    ----------
    trajectories : TrajectoryEnsemble
        Trajectories to estimate the timescales from.
    lag_time : float
        Lag time of the Koopman estimate. Must be an integer multiple of the
        ensemble's ``frame_stride``.
    feature_transform : FeatureTransform
        Transform applied to the frames of every trajectory.
    n_timescales : int | None
        Number of (slowest) timescales to return. By default all available
        ones, which is one per feature dimension - fewer if the estimated
        covariances are not of full rank.

    Returns
    -------
    np.ndarray
        Shape (n_timescales,). Timescales in the same time unit as
        ``frame_stride``, sorted from slow to fast.
    """
    frame_lag = resolve_frame_lag(trajectories, lag_time, name=name)
    featurized_trajs = featurize_trajectories(trajectories, feature_transform)

    vamp = dt.decomposition.VAMP(frame_lag).fit(featurized_trajs).fetch_model()
    timescales_in_frames = np.atleast_1d(vamp.timescales(k=n_timescales))

    return timescales_in_frames * trajectories.frame_stride
