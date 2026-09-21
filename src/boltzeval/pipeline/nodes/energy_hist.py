from matplotlib import pyplot as plt
import numpy as np
from boltzeval.metrics.energy_hist import get_energy_hist, determine_energy_hist_range
from boltzeval.metrics.hist_comparison import (
    HistogramMetric,
    get_hist_jensen_shannon,
)
from boltzeval.pipeline.eval import EvaluationNode
from boltzeval.utils.hist_visualization import (
    plot_as_density,
    visualize_histogram_1d,
    visualize_histogram_1d_dual,
)


ENERGY_LABEL = r"energy / $k_B T$"


class EnergyHistNode(EvaluationNode):
    """
    Compares the target energy histogram of predicted against reference samples.
    """

    requirements = ["true_samples_target_log_prob", "pred_samples_target_log_prob"]

    def __init__(
        self,
        include_pdf: bool = True,
        include_pred_histogram: bool = True,
        include_true_histogram: bool = True,
        hist_metrics: list[HistogramMetric] | None = None,
        energy_range: tuple[float, float] | None = None,
        ignore_pred_range: bool = False,
    ):
        super().__init__()
        self.include_pdf = include_pdf
        self.include_pred_histogram = include_pred_histogram
        self.include_true_histogram = include_true_histogram
        self.hist_metrics = hist_metrics if hist_metrics is not None else []
        self.energy_range = energy_range
        self.ignore_pred_range = ignore_pred_range

    def _eval(self, data):
        metrics = {}

        if self.energy_range is None:
            if data.true_samples_target_log_prob.shape[0] > 0:
                true_range = determine_energy_hist_range(
                    data.true_samples_target_log_prob
                )
            else:
                true_range = determine_energy_hist_range(
                    data.pred_samples_target_log_prob
                )

            if self.ignore_pred_range:
                energy_range = true_range
            else:
                pred_range = determine_energy_hist_range(
                    data.pred_samples_target_log_prob
                )
                energy_range = (
                    min(true_range[0], pred_range[0]),
                    max(true_range[1], pred_range[1]),
                )
        else:
            energy_range = self.energy_range

        # === Get true and predicted energy histogram ===
        true_energy_hist = get_energy_hist(
            data.true_samples_target_log_prob, energy_range=energy_range
        )
        pred_energy_hist = get_energy_hist(
            data.pred_samples_target_log_prob, energy_range=energy_range
        )

        # === Optionally log true and predicted hist ===
        if self.include_true_histogram:
            metrics["energy_hist/true_hist"] = true_energy_hist

        if self.include_pred_histogram:
            metrics["energy_hist/pred_hist"] = pred_energy_hist

        # === Optionally log pdf visualization of true and predicted hist ===
        if self.include_pdf:
            energy_hist_pdf = visualize_histogram_1d_dual(
                true_hist=true_energy_hist,
                pred_hist=pred_energy_hist,
                vis_mode=plot_as_density,
                xlabel=ENERGY_LABEL,
                ylabel="density",
            )
            metrics["energy_hist/pdf"] = energy_hist_pdf

        # === Compute comparison metrics between both histograms ===
        # Note that this is after discretization as histograms and
        # not on the samples used to build that histogram!
        for hist_metric in self.hist_metrics:
            m = hist_metric(true_energy_hist, pred_energy_hist)
            metrics[f"energy_hist/{hist_metric.id}"] = m

        return metrics


class EnergyHistNodeSingle(EvaluationNode):
    """
    Target energy histogram of a single set of samples.

    The single-dataset counterpart of :class:`EnergyHistNode`: it visualizes
    the energies of the reference samples on their own, without anything to
    compare them against, so no histogram comparison metrics are produced.
    """

    requirements = ["true_samples_target_log_prob"]

    def __init__(
        self,
        include_pdf: bool = True,
        include_histogram: bool = False,
        energy_range: tuple[float, float] | None = None,
    ):
        """
        Parameters
        ----------
        include_pdf : bool
            Whether to produce the visualization of the energy histogram.
        include_histogram : bool
            Whether to also return the raw histogram.
        energy_range : tuple[float, float] | None
            Explicit histogram range. Determined from the energies if None.
        """
        super().__init__()
        self.include_pdf = include_pdf
        self.include_histogram = include_histogram
        self.energy_range = energy_range

    def _eval(self, data):
        metrics = {}

        log_probs = data.true_samples_target_log_prob

        energy_range = self.energy_range
        if energy_range is None:
            energy_range = determine_energy_hist_range(log_probs)

        energy_hist = get_energy_hist(log_probs, energy_range=energy_range)

        if self.include_histogram:
            metrics["energy_hist/hist"] = energy_hist

        if self.include_pdf:
            metrics["energy_hist/pdf"] = visualize_histogram_1d(
                energy_hist,
                vis_mode=plot_as_density,
                xlabel=ENERGY_LABEL,
                ylabel="density",
            )

        return metrics


if __name__ == "__main__":
    from boltzeval.utils.pdf import plot_pdf
    from boltzeval.pipeline.eval import get_pdfs, run_eval, EvalData

    rng = np.random.default_rng(seed=42)

    # -------------------------
    # Generate dummy samples
    # -------------------------
    batch_size = 100_000
    dim = 2

    # True samples ~ N(0, I)
    samples_true = rng.normal(loc=0.0, scale=1.0, size=(batch_size, dim))

    # Pred samples ~ N(0.5, 1.2 I)
    pred_scale = 1.2
    pred_mean = 0.01
    samples_pred = rng.normal(loc=pred_mean, scale=pred_scale, size=(batch_size, dim))

    # -------------------------
    # Dummy log-probabilities
    # -------------------------
    # Target log-prob (assume true distribution N(0, I))
    true_samples_target_log_prob = -0.5 * np.sum(samples_true**2, axis=1)
    pred_samples_target_log_prob = -0.5 * np.sum(samples_pred**2, axis=1)

    # Model log-prob (assume model N(0.5, 1.2 I))
    true_samples_model_log_prob = -0.5 * np.sum(
        ((samples_true - pred_mean) / pred_scale) ** 2, axis=1
    )
    pred_samples_model_log_prob = -0.5 * np.sum(
        ((samples_pred - pred_mean) / pred_scale) ** 2, axis=1
    )

    data = EvalData(
        samples_true=samples_true,
        samples_pred=samples_pred,
        true_samples_target_log_prob=true_samples_target_log_prob,
        pred_samples_target_log_prob=pred_samples_target_log_prob,
        true_samples_model_log_prob=true_samples_model_log_prob,
        pred_samples_model_log_prob=pred_samples_model_log_prob,
    )

    pipeline = [EnergyHistNode(hist_metrics=[get_hist_jensen_shannon])]

    # -------------------------
    # Run evaluation
    # -------------------------
    metrics = run_eval(data, pipeline=pipeline)

    # -------------------------
    # Print results
    # -------------------------
    print("\n=== Evaluation Metrics ===\n")

    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            print(f"{k:30s}: {v:.6f}")
        else:
            print(f"{k:30s}: {v}")

    pdfs = get_pdfs(metrics)
    for pdf in pdfs.values():
        print(pdf)
        plot_pdf(pdf, show=True)

    print("\nDone.")
