import numpy as np
import deeptime as dt
from deeptime.base import Transformer

from boltzeval.metrics.feature_transforms import (
    FeatureTransform,
    featurize_trajectories,
)
from boltzeval.utils.histogram import Histogram
from boltzeval.utils.trajectory import TrajectoryEnsemble, resolve_frame_lag


def fit_tica(
    trajectories: TrajectoryEnsemble,
    lag_time: float,
    feature_transform: FeatureTransform,
    dim: int | None = 2,
    use_koopman=False,
):
    """
    Fit a TICA model from a trajectory ensemble, a physical lag time and a
    feature transform.

    Parameters
    ----------
    trajectories : TrajectoryEnsemble
        Trajectories to fit on. Their ``frame_stride`` translates ``lag_time``
        into the frame lag the estimator works with.
    lag_time : float
        Lag time in the unit of the ensemble's ``frame_stride``. Must be an
        integer multiple of it.
    feature_transform : FeatureTransform
        Transform applied to the frames of every trajectory.
    dim : int | None
        Number of TICA components to keep.
    use_koopman : bool
        Whether to reweight non-equilibrium data with a Koopman estimator.
    """
    frame_lag = resolve_frame_lag(trajectories, lag_time)
    featurized_trajs = featurize_trajectories(trajectories, feature_transform)
    tica = dt.decomposition.TICA(dim=dim, lagtime=frame_lag)

    if use_koopman:
        koopman_estimator = dt.covariance.KoopmanWeightingEstimator(lagtime=lag)
        reweighting_model = koopman_estimator.fit(featurized_trajs).fetch_model()
        tica_model = tica.fit(featurized_trajs, reweighting_model).fetch_model()
    else:
        tica_model = tica.fit(featurized_trajs).fetch_model()

    return tica_model


def project_tica(
    samples: np.ndarray, tica: Transformer, feature_transform: FeatureTransform
):
    """
    Project sampels into TICA coordinate system using a pre-trained TICA model and a feature transform.
    """
    featurized_samples = feature_transform(samples)
    projections: np.ndarray = tica.transform(featurized_samples)
    return projections


def get_tica_hist(
    samples: np.ndarray,
    tica: Transformer,
    feature_transform: FeatureTransform,
    bins: int = 100,
    data_range: tuple | None = None,
):
    """
    Create TICA histogram from samples, a pre-trained TICA model and a feature transform.
    """
    projections = project_tica(
        samples=samples, tica=tica, feature_transform=feature_transform
    )
    return Histogram.from_samples(projections, bins=bins, data_range=data_range)


# vvvvvvvv Small demo for testing vvvvvvvv
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from boltzeval.utils.hist_visualization import (
        visualize_histogram_1d,
        visualize_histogram_2d,
        visualize_histogram_2d_dual,
    )
    from boltzeval.utils.hist_visualization import plot_as_free_energy
    from boltzeval.utils.histogram import Histogram

    np.random.seed(0)

    # -------------------------------------------------------
    # Multimodal potential: mixture of two quadratic wells
    # U(x) = min(||x - c1||^2, ||x - c2||^2)
    # smooth-ish version via softmin
    # -------------------------------------------------------

    c1 = np.array([-2.0, 0.0])
    c2 = np.array([2.0, 0.0])

    def grad_U(x):
        # soft assignment to wells (softmin)
        d1 = np.exp(-np.linalg.norm(x - c1) ** 2)
        d2 = np.exp(-np.linalg.norm(x - c2) ** 2)
        w1 = d1 / (d1 + d2)
        w2 = d2 / (d1 + d2)

        # gradient of quadratic wells weighted by soft assignments
        return w1 * (x - c1) + w2 * (x - c2)

    # Time between two saved frames of the simulation below
    time_step = 0.05

    def simulate(T=1000, time_step=time_step, sigma=0.4):
        x = np.zeros((T, 2))
        x[0] = np.random.randn(2)

        for t in range(1, T):
            noise = np.sqrt(time_step) * sigma * np.random.randn(2)
            x[t] = x[t - 1] - time_step * grad_U(x[t - 1]) + noise

        return x

    # -------------------------------------------------------
    # identity features
    # -------------------------------------------------------
    feature_transform = lambda traj: traj

    # -------------------------------------------------------
    # trajectories from multimodal system
    # -------------------------------------------------------
    trajectories = TrajectoryEnsemble.from_array(
        [simulate() for _ in range(5)], frame_stride=time_step
    )

    # -------------------------------------------------------
    # fit TICA
    # -------------------------------------------------------
    tica_model = fit_tica(
        trajectories=trajectories,
        lag_time=10 * time_step,
        feature_transform=feature_transform,
        dim=2,
    )

    # -------------------------------------------------------
    # project
    # -------------------------------------------------------
    samples = trajectories.as_samples()
    hist = get_tica_hist(samples, tica_model, feature_transform)

    if hist.ndim == 1:
        visualize_histogram_1d(hist, vis_mode=plot_as_free_energy, show=True)
    else:
        visualize_histogram_2d(hist, vis_mode=plot_as_free_energy, show=True)
