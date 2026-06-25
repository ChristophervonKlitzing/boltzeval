import numpy as np
import deeptime as dt
from deeptime.base import Transformer

from boltzeval.metrics.feature_transforms import FeatureTransform
from boltzeval.utils.histogram import Histogram


def fit_tica(
    trajectories: list[np.ndarray] | np.ndarray,
    lag: int,
    feature_transform: FeatureTransform,
    dim: int | None = 2,
    use_koopman=False,
):
    """
    Fits a TICA model from a list of trajectories, a lag time and a feature transform.
    trajectories must be either a list of arrays each of shape (#frames, dim)
    or one array of shape (#trajectories, #frames, dim)
    """
    featurized_trajs = np.stack([feature_transform(t) for t in trajectories])
    tica = dt.decomposition.TICA(dim=dim, lagtime=lag)

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

    def simulate(T=1000, dt=0.05, sigma=0.4):
        x = np.zeros((T, 2))
        x[0] = np.random.randn(2)

        for t in range(1, T):
            noise = np.sqrt(dt) * sigma * np.random.randn(2)
            x[t] = x[t - 1] - dt * grad_U(x[t - 1]) + noise

        return x

    # -------------------------------------------------------
    # identity features
    # -------------------------------------------------------
    feature_transform = lambda traj: traj

    # -------------------------------------------------------
    # trajectories from multimodal system
    # -------------------------------------------------------
    trajectories = [simulate() for _ in range(5)]

    # -------------------------------------------------------
    # fit TICA
    # -------------------------------------------------------
    tica_model = fit_tica(
        trajectories=trajectories,
        lag=10,
        feature_transform=feature_transform,
        dim=2,
    )

    # -------------------------------------------------------
    # project
    # -------------------------------------------------------
    samples = np.concatenate(trajectories, 0)
    hist = get_tica_hist(samples, tica_model, feature_transform)

    if hist.ndim == 1:
        visualize_histogram_1d(hist, vis_mode=plot_as_free_energy, show=True)
    else:
        visualize_histogram_2d(hist, vis_mode=plot_as_free_energy, show=True)
