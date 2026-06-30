import mdtraj as md
import numpy as np

from boltzeval.metrics.hist_comparison import (
    HistogramMetric,
    eval_histogram_metrics,
    get_hist_jensen_shannon,
)
from boltzeval.metrics.torsion_marginals import (
    get_free_energy_difference,
    get_torsion_angles,
    get_torsion_marginal_hists,
    visualize_torsion_marginals_dual,
)
from boltzeval.pipeline.eval import EvaluationNode
from boltzeval.utils.hist_visualization import VisualizationMode, plot_as_log_density
from boltzeval.utils.histogram import Histogram


class TorsionMarginalEval(EvaluationNode):
    requirements = ["samples_pred", "samples_true"]

    def __init__(
        self,
        topology: md.Topology,
        vis_mode: VisualizationMode = plot_as_log_density,
        include_pdf: bool = True,
        include_true_histograms: bool = True,
        include_pred_histograms: bool = True,
        include_free_energy_difference: bool = False,
        histogram_metrics: tuple[HistogramMetric] = [get_hist_jensen_shannon],
    ):
        super().__init__()
        self._topology = topology
        self.vis_mode = vis_mode

        self.include_pdf = include_pdf
        self.include_true_histograms = include_true_histograms
        self.include_pred_histograms = include_pred_histograms
        self.include_free_energy_difference = include_free_energy_difference

        self.histogram_metrics = histogram_metrics

    def _eval(self, data):
        samples_true = self._reshape(data.samples_true)
        samples_pred = self._reshape(data.samples_pred)

        torsion_metrics = self._get_torsion_marginal_metrics(
            samples_true=samples_true, samples_pred=samples_pred
        )
        return torsion_metrics

    @staticmethod
    def _reshape(obj: np.ndarray):
        return obj.reshape(obj.shape[0], -1, 3)

    def _get_torsion_marginal_metrics(
        self, samples_true: np.ndarray, samples_pred: np.ndarray
    ):
        metrics = {}
        angles_true = get_torsion_angles(samples_true, self._topology)
        angles_pred = get_torsion_angles(samples_pred, self._topology)

        torsion_marginals_true = get_torsion_marginal_hists(*angles_true)
        torsion_marginals_pred = get_torsion_marginal_hists(*angles_pred)

        if self.include_free_energy_difference:
            # Compute free energy difference on phi angles
            # (neg log weight ratio between high and low energy region)
            phis_true = angles_true[0]
            phis_pred = angles_pred[0]
            free_energy_difference_true = get_free_energy_difference(phis_true)
            free_energy_difference_pred = get_free_energy_difference(phis_pred)
            metrics["torsion_marginals/free_energy_difference_true"] = (
                free_energy_difference_true
            )
            metrics["torsion_marginals/free_energy_difference_pred"] = (
                free_energy_difference_pred
            )

        # Compute histogram metrics
        if len(self.histogram_metrics) > 0:
            # Only get 2D histogram metrics fow now
            marginals_true_2d = torsion_marginals_true[0]
            marginals_pred_2d = torsion_marginals_pred[0]

            metrics.update(
                eval_histogram_metrics(
                    self.histogram_metrics,
                    marginals_true_2d,
                    marginals_pred_2d,
                    group="torsion_marginals",
                    hist_type="phi_psi",
                ),
            )

        if self.include_pdf:
            pdf_buffer = visualize_torsion_marginals_dual(
                torsion_marginals_true=torsion_marginals_true,
                torsion_marginals_pred=torsion_marginals_pred,
                vis_mode=self.vis_mode,
            )

            infill = self.vis_mode.id
            key = f"torsion_marginals/{infill}_pdf"
            metrics[key] = pdf_buffer

        def flatten_marginals(
            marginals: tuple[list[Histogram], list[Histogram], list[Histogram]],
            prefix: str,
        ):
            d: dict[str, Histogram] = {}
            for i, (ram_hist, phi_hist, psi_hist) in enumerate(zip(*marginals)):
                key_part = f"torsion_marginals/{prefix}_hist_{i}"
                d[f"{key_part}_phi_psi"] = ram_hist
                d[f"{key_part}_phi"] = phi_hist
                d[f"{key_part}_psi"] = psi_hist
            return d

        if self.include_true_histograms:
            flattened_marginals_true = flatten_marginals(torsion_marginals_true, "true")
            metrics.update(flattened_marginals_true)

        if self.include_pred_histograms:
            flattened_marginals_pred = flatten_marginals(torsion_marginals_pred, "pred")
            metrics.update(flattened_marginals_pred)

        return metrics
