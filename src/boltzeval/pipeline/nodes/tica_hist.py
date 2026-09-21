from deeptime.base import Transformer

from boltzeval.metrics.feature_transforms import FeatureTransform
from boltzeval.metrics.tica import project_tica
from boltzeval.pipeline import EvaluationNode
from boltzeval.utils.hist_visualization import (
    VisualizationMode,
    plot_as_free_energy,
    visualize_histogram_2d,
    visualize_histogram_2d_dual,
)

from boltzeval.metrics.hist_comparison import HistogramMetric
from boltzeval.utils.histogram import Histogram
import numpy as np


class TicaHistNode(EvaluationNode):
    """
    Compares predicted against reference samples in TICA coordinates.
    """

    requirements = ["samples_true", "samples_pred"]

    def __init__(
        self,
        tica: Transformer,
        feature_transform: FeatureTransform,
        include_pdf: bool = True,
        include_pred_histogram: bool = True,
        include_true_histogram: bool = True,
        vis_mode: VisualizationMode = plot_as_free_energy,
        hist_metrics: list[HistogramMetric] | None = None,
        bins=100,
    ):
        super().__init__()

        self.include_pdf = include_pdf
        self.include_pred_histogram = include_pred_histogram
        self.include_true_histogram = include_true_histogram

        self._tica = tica
        self._feature_transform = feature_transform
        self._vis_mode = vis_mode
        self._hist_metrics = hist_metrics if hist_metrics is not None else []
        self._bins = bins

    def _eval(self, data):
        metrics = {}

        samples_true = data.samples_true
        samples_pred = data.samples_pred

        projections_true = project_tica(
            samples_true, self._tica, self._feature_transform
        )
        projections_pred = project_tica(
            samples_pred, self._tica, self._feature_transform
        )
        projections_total = np.concatenate([projections_true, projections_pred])
        assert projections_total.shape[1] == 2

        x_min = projections_total[:, 0].min()
        x_max = projections_total[:, 0].max()
        y_min = projections_total[:, 1].min()
        y_max = projections_total[:, 1].max()

        data_range = (x_min, x_max, y_min, y_max)

        true_hist = Histogram.from_samples(
            projections_true, bins=self._bins, data_range=data_range
        )

        pred_hist = Histogram.from_samples(
            projections_pred, bins=self._bins, data_range=data_range
        )

        if self.include_pdf:
            # Visualize histogram -> TICA plot
            pdf = visualize_histogram_2d_dual(
                true_hist=true_hist, pred_hist=pred_hist, vis_mode=self._vis_mode
            )
            metrics[f"tica/pdf"] = pdf

        # Compute histogram metrics
        for hist_metric in self._hist_metrics:
            m = hist_metric(true_hist, pred_hist)
            metrics[f"tica/{hist_metric.id}"] = m

        if self.include_true_histogram:
            metrics[f"tica/true_hist"] = true_hist

        if self.include_pred_histogram:
            metrics[f"tica/pred_hist"] = pred_hist

        return metrics


class TicaHistNodeSingle(EvaluationNode):
    """
    Histogram of a single set of samples in TICA coordinates.

    The single-dataset counterpart of :class:`TicaHistNode`: it projects the
    reference samples with a pre-trained TICA model and visualizes them on
    their own, so no histogram comparison metrics are produced.
    """

    requirements = ["samples_true"]

    def __init__(
        self,
        tica: Transformer,
        feature_transform: FeatureTransform,
        include_pdf: bool = True,
        include_histogram: bool = True,
        include_projections: bool = False,
        vis_mode: VisualizationMode = plot_as_free_energy,
        bins=100,
        data_range: tuple[float, float, float, float] | None = None,
    ):
        """
        Parameters
        ----------
        tica : Transformer
            Pre-trained TICA model the samples are projected with.
        feature_transform : FeatureTransform
            Transform applied to the samples before the projection.
        include_pdf : bool
            Whether to produce the visualization of the TICA histogram.
        include_histogram : bool
            Whether to also return the raw histogram.
        include_projections : bool
            Whether to also return the raw TICA projections of the samples,
            of shape (n_samples, 2).
        vis_mode : VisualizationMode
            How the histogram is mapped to plotted values.
        bins : int
            Number of histogram bins per dimension.
        data_range : tuple[float, float, float, float] | None
            Explicit (x_min, x_max, y_min, y_max) histogram range. Determined
            from the projections if None.
        """
        super().__init__()

        self.include_pdf = include_pdf
        self.include_histogram = include_histogram
        self.include_projections = include_projections

        self._tica = tica
        self._feature_transform = feature_transform
        self._vis_mode = vis_mode
        self._bins = bins
        self._data_range = data_range

    def _eval(self, data):
        metrics = {}

        projections = project_tica(
            data.samples_true, self._tica, self._feature_transform
        )
        assert projections.shape[1] == 2

        hist = Histogram.from_samples(
            projections, bins=self._bins, data_range=self._data_range
        )

        if self.include_projections:
            metrics["tica/projections"] = projections

        if self.include_histogram:
            metrics["tica/hist"] = hist

        if self.include_pdf:
            metrics["tica/pdf"] = visualize_histogram_2d(
                hist, vis_mode=self._vis_mode
            )

        return metrics


# vvvvvvvv Small demo for testing vvvvvvvv
if __name__ == "__main__":
    from boltzeval.metrics.tica import fit_tica
    from boltzeval.pipeline import EvalData, TrajectoryEnsemble
    from boltzeval.utils.hist_visualization import (
        visualize_histogram_2d_dual,
    )
    from boltzeval.metrics.hist_comparison import get_hist_jensen_shannon
    from boltzeval.utils.hist_visualization import plot_as_free_energy

    from boltzeval.utils.pdf import plot_pdf

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

    eval_node = TicaHistNode(
        tica_model, feature_transform, hist_metrics=[get_hist_jensen_shannon]
    )

    data = EvalData(
        samples_true=trajectories.as_samples(),
        samples_pred=trajectories[0].frames,
    )
    metrics = eval_node.eval(data)
    print(metrics)
    plot_pdf(metrics["tica/pdf"], show=True)
