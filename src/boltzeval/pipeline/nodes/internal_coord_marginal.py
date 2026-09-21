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
    visualize_torsion_marginals,
    visualize_torsion_marginals_dual,
)
from boltzeval.pipeline.eval import EvaluationNode
from boltzeval.utils.hist_visualization import VisualizationMode, plot_as_log_density
from boltzeval.utils.histogram import Histogram
from boltzeval.utils.shape_utils import reshape_to_molecular


def _flatten_marginals(
    marginals: tuple[list[Histogram], list[Histogram], list[Histogram]],
    prefix: str,
) -> dict[str, Histogram]:
    """
    Flatten the per-torsion-pair histograms into one dict of metric keys.
    """
    d: dict[str, Histogram] = {}
    for i, (ram_hist, phi_hist, psi_hist) in enumerate(zip(*marginals)):
        key_part = f"torsion_marginals/{prefix}hist_{i}"
        d[f"{key_part}_phi_psi"] = ram_hist
        d[f"{key_part}_phi"] = phi_hist
        d[f"{key_part}_psi"] = psi_hist
    return d


class TorsionMarginalNode(EvaluationNode):
    """
    Compares the backbone torsion marginals of predicted against reference samples.
    """

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
        samples_true = reshape_to_molecular(data.samples_true)
        samples_pred = reshape_to_molecular(data.samples_pred)

        torsion_metrics = self._get_torsion_marginal_metrics(
            samples_true=samples_true, samples_pred=samples_pred
        )
        return torsion_metrics

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

        if self.include_true_histograms:
            metrics.update(_flatten_marginals(torsion_marginals_true, "true_"))

        if self.include_pred_histograms:
            metrics.update(_flatten_marginals(torsion_marginals_pred, "pred_"))

        return metrics


class TorsionMarginalNodeSingle(EvaluationNode):
    """
    Backbone torsion marginals of a single set of samples.

    The single-dataset counterpart of :class:`TorsionMarginalNode`: it
    visualizes the phi/psi marginals of the reference samples on their own, so
    no histogram comparison metrics are produced.
    """

    requirements = ["samples_true"]

    def __init__(
        self,
        topology: md.Topology,
        vis_mode: VisualizationMode = plot_as_log_density,
        include_pdf: bool = True,
        include_histograms: bool = False,
        include_free_energy_difference: bool = False,
    ):
        """
        Parameters
        ----------
        topology : md.Topology
            Topology describing the molecular system of the samples.
        vis_mode : VisualizationMode
            How the histograms are mapped to plotted values.
        include_pdf : bool
            Whether to produce the visualization of the torsion marginals.
        include_histograms : bool
            Whether to also return the raw histograms.
        include_free_energy_difference : bool
            Whether to also report the free energy difference between the two
            phi regions, which is a single-dataset quantity.
        """
        super().__init__()
        self._topology = topology
        self.vis_mode = vis_mode

        self.include_pdf = include_pdf
        self.include_histograms = include_histograms
        self.include_free_energy_difference = include_free_energy_difference

    def _eval(self, data):
        metrics = {}

        samples = reshape_to_molecular(data.samples_true)

        angles = get_torsion_angles(samples, self._topology)
        torsion_marginals = get_torsion_marginal_hists(*angles)

        if self.include_free_energy_difference:
            # Computed on the phi angles
            # (neg log weight ratio between high and low energy region)
            phis = angles[0]
            metrics["torsion_marginals/free_energy_difference"] = (
                get_free_energy_difference(phis)
            )

        if self.include_pdf:
            pdf_buffer = visualize_torsion_marginals(
                torsion_marginals=torsion_marginals,
                vis_mode=self.vis_mode,
            )

            infill = self.vis_mode.id
            metrics[f"torsion_marginals/{infill}_pdf"] = pdf_buffer

        if self.include_histograms:
            metrics.update(_flatten_marginals(torsion_marginals, ""))

        return metrics
