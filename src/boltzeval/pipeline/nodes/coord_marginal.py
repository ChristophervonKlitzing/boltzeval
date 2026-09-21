import itertools
from typing import Callable

from boltzeval.metrics.hist_comparison import (
    eval_histogram_metrics,
    get_hist_jensen_shannon,
)
from boltzeval.pipeline import EvaluationNode
import numpy as np

from boltzeval.utils.hist_visualization import (
    VisualizationMode,
    plot_as_log_density,
    visualize_histogram_1d,
    visualize_histogram_1d_dual,
    visualize_histogram_2d,
    visualize_histogram_2d_dual,
)
from boltzeval.utils.histogram import Histogram


def _get_all_subsets(n: int) -> list[tuple[int, ...]]:
    """
    Return all non-empty subsets of range(n); e.g. n=3 -> (0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2).
    """
    result = []
    for r in range(1, n + 1):
        result.extend(itertools.combinations(range(n), r))
    return result


def _list_to_str(l: list) -> str:
    return "_".join([str(e) for e in l])


def _resolve_marginals(
    marginals: list[tuple[int, ...]] | None, dim: int
) -> list[tuple[int, ...]]:
    """
    The requested marginals, or all non-empty coordinate subsets if unspecified.
    """
    if marginals is None:
        return _get_all_subsets(dim)
    return marginals


class CoordinateMarginalNode(EvaluationNode):
    """
    Compares the coordinate marginals of predicted against reference samples.
    """

    requirements = ["samples_pred", "samples_true"]

    def __init__(
        self,
        n_bins: int = 50,
        hist_metrics=[get_hist_jensen_shannon],
        group="coordinate_marginals",
        sample_transform: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        transform_label: str = "",
        marginals: list[tuple[int, ...]] | None = None,
        include_pred_histograms: bool = True,
        include_true_histograms: bool = True,
        include_pdfs: bool = True,
        vis_mode: VisualizationMode = plot_as_log_density,
    ):
        super().__init__()

        self._n_bins = n_bins
        self._hist_metrics = hist_metrics
        self._group = group
        self._sample_transform = sample_transform
        self._transform_label = transform_label
        self._marginals = marginals
        self._vis_mode = vis_mode

        self._include_pred_hists = include_pred_histograms
        self._include_true_hists = include_true_histograms
        self._include_pdfs = include_pdfs

    def _get_marginals(self, dim: int) -> list[tuple[int, ...]]:
        return _resolve_marginals(self._marginals, dim)

    def _eval(self, data):
        metrics = {}

        # Retrieve and transform/project samples (e.g., a TICA transform)
        samples_true = self._sample_transform(data.samples_true)
        samples_pred = self._sample_transform(data.samples_pred)

        dim = samples_true.shape[1]

        for marginal_coords in self._get_marginals(dim):
            data_range = None

            if samples_true.shape[0] > 0:
                hist_true = Histogram.from_samples(
                    samples_true[:, marginal_coords], bins=self._n_bins
                )
                data_range = hist_true.get_support_range()
                hist_pred = Histogram.from_samples(
                    samples_pred[:, marginal_coords],
                    bins=self._n_bins,
                    data_range=data_range,
                )
            else:
                hist_pred = Histogram.from_samples(
                    samples_pred[:, marginal_coords],
                    bins=self._n_bins,
                )
                data_range = hist_pred.get_support_range()
                hist_true = Histogram.from_samples(
                    samples_true[:, marginal_coords],
                    bins=self._n_bins,
                    data_range=data_range,
                )

            metrics.update(
                self._log_histograms(
                    marginal_coords=marginal_coords,
                    hist_true=hist_true,
                    hist_pred=hist_pred,
                )
            )

            metrics.update(
                self._visualize_histograms(
                    marginal_coords=marginal_coords,
                    hist_true=hist_true,
                    hist_pred=hist_pred,
                )
            )

            metrics.update(
                self._compute_histogram_metrics(
                    marginal_coords=marginal_coords,
                    hist_true=hist_true,
                    hist_pred=hist_pred,
                )
            )

        return metrics

    def _log_histograms(
        self,
        marginal_coords: tuple[int, ...],
        hist_true: Histogram,
        hist_pred: Histogram,
    ):
        metrics = {}
        if self._include_true_hists:
            metrics[f"{self._group}/hist_true_{_list_to_str(marginal_coords)}"] = (
                hist_true
            )

        if self._include_pred_hists:
            metrics[f"{self._group}/hist_pred_{_list_to_str(marginal_coords)}"] = (
                hist_pred
            )

        return metrics

    def _visualize_histograms(
        self,
        marginal_coords: tuple[int, ...],
        hist_true: Histogram,
        hist_pred: Histogram,
    ):
        metrics = {}

        label = f"{self._transform_label}{_list_to_str(marginal_coords)}"
        hist_dim = len(marginal_coords)
        if hist_dim == 1:
            pdf = visualize_histogram_1d_dual(
                true_hist=hist_true,
                pred_hist=hist_pred,
                vis_mode=self._vis_mode,
                xlabel=label,
            )
            metrics[f"{self._group}/marginal_{label}_pdf"] = pdf

        elif hist_dim == 2:
            pdf = visualize_histogram_2d_dual(
                true_hist=hist_true, pred_hist=hist_pred, vis_mode=self._vis_mode
            )
            metrics[f"{self._group}/marginal_{label}_pdf"] = pdf
        else:
            # Don't visualize higher-dimensional histograms for now
            pass

        return metrics

    def _compute_histogram_metrics(
        self,
        marginal_coords: tuple[int, ...],
        hist_true: Histogram,
        hist_pred: Histogram,
    ):
        label = _list_to_str(marginal_coords)
        return eval_histogram_metrics(
            hist_metrics=self._hist_metrics,
            true=[hist_true],
            pred=[hist_pred],
            group=self._group,
            hist_type=f"marginal_{label}",
            include_aggregated=False,  # Makes no sense with just one histogram
        )


class CoordinateMarginalNodeSingle(EvaluationNode):
    """
    Coordinate marginals of a single set of samples.

    The single-dataset counterpart of :class:`CoordinateMarginalNode`: it
    visualizes the marginals of the reference samples on their own, so no
    histogram comparison metrics are produced.
    """

    requirements = ["samples_true"]

    def __init__(
        self,
        n_bins: int = 50,
        group: str = "coordinate_marginals",
        sample_transform: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        transform_label: str = "",
        marginals: list[tuple[int, ...]] | None = None,
        include_histograms: bool = True,
        include_pdfs: bool = True,
        vis_mode: VisualizationMode = plot_as_log_density,
    ):
        """
        Parameters
        ----------
        n_bins : int
            Number of histogram bins per dimension.
        group : str
            Prefix of the produced metric keys.
        sample_transform : Callable[[np.ndarray], np.ndarray]
            Projection applied to the samples before the marginals are taken.
        transform_label : str
            Prefix of the axis labels, e.g. the name of the projection.
        marginals : list[tuple[int, ...]] | None
            Coordinate subsets to marginalize onto. All non-empty subsets if
            None, which grows quickly with the dimension.
        include_histograms : bool
            Whether to also return the raw histograms.
        include_pdfs : bool
            Whether to produce the visualizations (1D and 2D marginals only).
        vis_mode : VisualizationMode
            How the histograms are mapped to plotted values.
        """
        super().__init__()

        self._n_bins = n_bins
        self._group = group
        self._sample_transform = sample_transform
        self._transform_label = transform_label
        self._marginals = marginals
        self._vis_mode = vis_mode

        self._include_hists = include_histograms
        self._include_pdfs = include_pdfs

    def _eval(self, data):
        metrics = {}

        # Retrieve and transform/project samples (e.g., a TICA transform)
        samples = self._sample_transform(data.samples_true)
        dim = samples.shape[1]

        for marginal_coords in _resolve_marginals(self._marginals, dim):
            hist = Histogram.from_samples(
                samples[:, marginal_coords], bins=self._n_bins
            )

            label = _list_to_str(marginal_coords)

            if self._include_hists:
                metrics[f"{self._group}/hist_{label}"] = hist

            if self._include_pdfs:
                metrics.update(
                    self._visualize_histogram(
                        marginal_coords=marginal_coords, hist=hist
                    )
                )

        return metrics

    def _visualize_histogram(self, marginal_coords: tuple[int, ...], hist: Histogram):
        metrics = {}

        def coord_label(coord: int) -> str:
            return f"{self._transform_label}{coord}"

        label = f"{self._transform_label}{_list_to_str(marginal_coords)}"
        hist_dim = len(marginal_coords)

        if hist_dim == 1:
            pdf = visualize_histogram_1d(
                hist, vis_mode=self._vis_mode, xlabel=label
            )
            metrics[f"{self._group}/marginal_{label}_pdf"] = pdf

        elif hist_dim == 2:
            pdf = visualize_histogram_2d(
                hist,
                vis_mode=self._vis_mode,
                xlabel=coord_label(marginal_coords[0]),
                ylabel=coord_label(marginal_coords[1]),
            )
            metrics[f"{self._group}/marginal_{label}_pdf"] = pdf
        else:
            # Don't visualize higher-dimensional histograms for now
            pass

        return metrics


if __name__ == "__main__":
    print(_get_all_subsets(3))
