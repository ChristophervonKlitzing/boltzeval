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
    visualize_histogram_1d_dual,
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


class CoordinateMarginalEval(EvaluationNode):
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
        if self._marginals is None:
            return _get_all_subsets(dim)
        else:
            return self._marginals

    def _eval(self, data):
        metrics = {}

        # Retrieve and transform/project samples (e.g., a TICA transform)
        samples_true = self._sample_transform(data.samples_true)
        samples_pred = self._sample_transform(data.samples_pred)

        dim = samples_true.shape[1]

        for marginal_coords in self._get_marginals(dim):
            hist_true = Histogram.from_samples(
                samples_true[:, marginal_coords], bins=self._n_bins
            )
            true_range = hist_true.get_support_range()

            hist_pred = Histogram.from_samples(
                samples_pred[:, marginal_coords],
                bins=self._n_bins,
                data_range=true_range,
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


if __name__ == "__main__":
    print(_get_all_subsets(3))
