from typing import Callable

from matplotlib import pyplot as plt
import numpy as np

from boltzeval.pipeline import EvaluationNode
from boltzeval.utils.hist_visualization import VisualizationMode, plot_as_log_density
from boltzeval.utils.pdf import matplotlib_to_pdf_buffer
from boltzeval.utils.plot import plot_2d


class SamplePlot2DEval(EvaluationNode):
    requirements = ["samples_true", "samples_pred"]

    def __init__(
        self,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        target_log_prob_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        log_prob_range: tuple[float, float] | None = None,
        key: str = "Media/sample_plot_2d",
        n_max_scatter_samples: int = 200,
        xlabel: str = "x",
        ylabel: str = "y",
        true_label="true",
        pred_label="pred",
        cmap: str = "viridis",
    ):
        super().__init__()

        self._xlim = xlim
        self._ylim = ylim
        self._target_log_prob_fn = target_log_prob_fn
        self._log_prob_range = log_prob_range
        self._key = key
        self._n_max_scatter_samples = n_max_scatter_samples

        self._xlabel = xlabel
        self._ylabel = ylabel

        self._true_label = true_label
        self._pred_label = pred_label

        self._cmap = cmap

    def _eval(self, data):
        samples_true = data.samples_true
        samples_pred = data.samples_pred

        assert samples_true.ndim == 2
        assert samples_pred.ndim == 2
        assert samples_true.shape[1] == 2
        assert samples_pred.shape[1] == 2

        fig, ax = plt.subplots(figsize=(6, 5))

        xlim = self._xlim
        ylim = self._ylim

        if xlim is None:
            xlim = (samples_true[:, 0].min(), samples_true[:, 0].max())
        if ylim is None:
            ylim = (samples_true[:, 1].min(), samples_true[:, 1].max())

        if self._target_log_prob_fn is not None:
            plot_2d(
                self._target_log_prob_fn,
                xlim=xlim,
                ylim=ylim,
                log_prob_range=self._log_prob_range,
                ax=ax,
                cmap=self._cmap,
            )

        samples_true = samples_true[: self._n_max_scatter_samples]
        samples_pred = samples_pred[: self._n_max_scatter_samples]

        ax.scatter(
            samples_true[:, 0],
            samples_true[:, 1],
            s=4,
            c="crimson",
            alpha=0.5,
            marker="x",
            label=self._true_label,
        )

        ax.scatter(
            samples_pred[:, 0],
            samples_pred[:, 1],
            s=4,
            c="black",
            alpha=0.5,
            marker="x",
            label=self._pred_label,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(self._xlabel)
        ax.set_ylabel(self._ylabel)

        ax.legend()

        pdf = matplotlib_to_pdf_buffer(ax)
        plt.close()
        return {self._key: pdf}
