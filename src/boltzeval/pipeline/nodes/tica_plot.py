from deeptime.base import Transformer

from boltzeval.metrics.feature_transforms import FeatureTransform
from boltzeval.metrics.tica import project_tica
from boltzeval.pipeline import EvaluationNode
from boltzeval.utils.hist_visualization import (
    VisualizationMode,
    plot_as_free_energy,
    visualize_histogram_2d_dual,
)

from boltzeval.metrics.hist_comparison import HistogramMetric


class TicaPlot(EvaluationNode):
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
            metrics[f"tica/vis"] = pdf

        # Compute histogram metrics
        for hist_metric in self._hist_metrics:
            m = hist_metric(true_hist, pred_hist)
            metrics[f"tica/{hist_metric.id}"] = m

        if self.include_true_histogram:
            metrics[f"tica/true_hist"] = true_hist

        if self.include_pred_histogram:
            metrics[f"tica/pred_hist"] = pred_hist

        return metrics


# vvvvvvvv Small demo for testing vvvvvvvv
if __name__ == "__main__":
    import numpy as np
    from boltzeval.metrics.tica import fit_tica
    from boltzeval.pipeline import EvalData
    from boltzeval.utils.hist_visualization import (
        visualize_histogram_2d_dual,
    )
    from boltzeval.metrics.hist_comparison import get_hist_jensen_shannon
    from boltzeval.utils.hist_visualization import plot_as_free_energy
    from boltzeval.utils.histogram import Histogram
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

    eval_node = TicaPlot(
        tica_model, feature_transform, hist_metrics=[get_hist_jensen_shannon]
    )

    data = EvalData(
        samples_true=np.concatenate(trajectories),
        samples_pred=trajectories[0],
    )
    metrics = eval_node.eval(data)
    print(metrics)
    plot_pdf(metrics["tica/vis"], show=True)
