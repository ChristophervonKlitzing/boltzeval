from itertools import chain
from typing import Protocol

from matplotlib import pyplot as plt
import numpy as np

from boltzeval.utils.histogram import Histogram
from boltzeval.utils.conversion import to_free_energy as _to_free_energy
from boltzeval.utils.pdf import matplotlib_to_pdf_buffer


class VisualizationMode(Protocol):
    @property
    def id(self) -> str:
        pass

    def __call__(self, hist: Histogram) -> tuple[np.ndarray, str]:
        """
        Transforms the histogram counts from a histogram into y-values and a label
        for matplotlib plot (1D) or imshow (2D) given a histogram.
        """
        pass


def plot_as_free_energy(hist: Histogram) -> tuple[np.ndarray, str]:
    y = _to_free_energy(hist.get_normalized_counts())
    ylabel = r"free energy / $k_B T$"
    return y, ylabel


plot_as_free_energy.id = "free_energy"


def plot_as_density(hist: Histogram) -> tuple[np.ndarray, str]:
    y = hist.get_as_density()

    # This leaves areas without samples empty (no information).
    y[y == 0] = np.nan

    ylabel = "probability density"
    return y, ylabel


plot_as_density.id = "density"


def plot_as_log_density(hist: Histogram) -> tuple[np.ndarray, str]:
    density = hist.get_as_density()
    ylabel = "log density"

    mask = density > 0
    density[~mask] = 1e-300

    # Compute free energy for nonzero probabilities
    log_density = np.where(mask, np.log(density), -np.inf)

    return log_density, ylabel


plot_as_log_density.id = "log_density"


def visualize_histogram_1d(
    hist: Histogram,
    vis_mode: VisualizationMode = plot_as_log_density,
    show: bool = False,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    label: str | None = None,
    linestyle: str = "-",
    transpose: bool = False,
):
    """
    Plot a 1D histogram as counts or free energy.

    Parameters
    ----------
    hist : np.ndarray
        Histogram counts or probabilities. If plotting as free energy, these
        will be normalized internally. Shape: (n_bins,)

    bin_edges : np.ndarray
        Bin edges defining the histogram intervals. Length is one greater
        than the number of bins, i.e., shape: (n_bins + 1,). The i-th bin
        covers the interval [bin_edges[i], bin_edges[i+1]).
    vis_mode : VisualizationMode
        Transforms the given histogram counts/density into an np.ndarray for visualization.
        The function further returns default label, which is used if `ylabel` is None.
    show : bool
        Whether to immediately display the plot.
    ax : plt.Axes, optional
        Axes to plot on; if None, a new figure/axes is created.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for x-axis.
    ylabel : str, optional
        Label for y-axis.
    label : str, optional
        Legend label.
    linestyle : str
        Line style for the plot.

    Returns
    -------
    pdf_buffer : PdfBuffer | None
        Buffer containing the generated PDF or None if `ax` is not None
    """

    # Compute bin centers
    x = hist.get_bin_centers()[0]

    y, default_label = vis_mode(hist)

    if ylabel is None:
        ylabel = default_label

    # Create axes if not provided
    new_plot = ax is None
    if new_plot:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    if transpose:
        x, y = (y, x)
        xlabel, ylabel = (ylabel, xlabel)

    n_nans = np.isnan(y).astype(np.float32).sum()
    if n_nans == y.shape[0] - 1:  # all but one nan
        dirac_like = True
    else:
        dirac_like = False

    if dirac_like:
        dirac_idx = (~np.isnan(y)).astype(np.float32).argmax()
        offset = hist.get_bin_volume() / 2
        x_min = x[dirac_idx] - offset
        x_max = x[dirac_idx] + offset
        y_dirac_vals = [0, y[dirac_idx], 0]
        x_dirac_vals = [x_min, x[dirac_idx], x_max]
        ax.plot(x_dirac_vals, y_dirac_vals, label=label, linestyle=linestyle)
    else:
        # Plot the data
        ax.plot(x, y, label=label, linestyle=linestyle)

    # Add optional titles and labels
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if label:
        ax.legend()

    fig.tight_layout()

    if ax is None:
        pdf_buffer = matplotlib_to_pdf_buffer(fig)
    else:
        pdf_buffer = None

    # Show plot immediately if requested
    if show:
        plt.show()
    elif new_plot:
        plt.close(fig)

    return pdf_buffer


def visualize_histogram_2d(
    hist: Histogram,
    vis_mode: VisualizationMode = plot_as_log_density,
    show: bool = False,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: None | str = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar: bool = True,
    cbar_label: str | None = None,
):
    """
    Plot a 2D histogram as counts or free energy and return a PDF buffer.

    Parameters
    ----------
    hist : np.ndarray
        2D histogram counts or probabilities. Shape: (nx, ny)

    x_bin_edges : np.ndarray
        Bin edges for x-dimension. Shape: (nx + 1,)

    y_bin_edges : np.ndarray
        Bin edges for y-dimension. Shape: (ny + 1,)

    vis_mode : VisualizationMode
        Transforms the given histogram counts/density into an np.ndarray for visualization.
        The function further returns default label, which is used if `cbar_label` is None.

    show : bool
        Whether to immediately display the plot.

    ax : plt.Axes, optional
        Axes to plot on; if None, a new figure/axes is created.

    Returns
    -------
    pdf_buffer : PdfBuffer
        Buffer containing the generated PDF.
    """

    z, default_label = vis_mode(hist)
    if cbar_label is None:
        cbar_label = default_label

    # Figure ownership
    new_plot = ax is None
    if new_plot:
        if cbar:
            figsize = (6, 5)
        else:
            figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    extend = list(chain.from_iterable(hist.get_support_range()))
    # Plot
    im = ax.imshow(
        z.T,
        extent=extend,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # Labels and title
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Colorbar
    if cbar:
        cbar_obj = fig.colorbar(im, ax=ax)
        if cbar_label is not None:
            cbar_obj.set_label(cbar_label)

    fig.tight_layout()

    pdf_buffer = matplotlib_to_pdf_buffer(fig)

    if show:
        plt.show()
    elif new_plot:
        plt.close(fig)

    return pdf_buffer


def visualize_histogram_1d_dual(
    true_hist: Histogram,
    pred_hist: Histogram,
    vis_mode: VisualizationMode = plot_as_log_density,
    show: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
):
    if pred_hist.ndim != 1 or true_hist.ndim != 1:
        raise ValueError(
            f"Both histograms must be 1D but got {pred_hist.ndim}D and {true_hist.ndim}D"
        )

    fig, ax = plt.subplots()

    # Plot both on the same axis
    visualize_histogram_1d(
        true_hist,
        vis_mode=vis_mode,
        ax=ax,
        label="True",
        show=False,
    )

    visualize_histogram_1d(
        pred_hist,
        vis_mode=vis_mode,
        ax=ax,
        label="Pred",
        show=False,
    )

    ax.legend()

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    pdf_buffer = matplotlib_to_pdf_buffer(fig)

    if show:
        plt.show()
    else:
        plt.close()

    return pdf_buffer


def visualize_histogram_2d_dual(
    true_hist: Histogram,
    pred_hist: Histogram,
    vis_mode: VisualizationMode = plot_as_log_density,
    show: bool = False,
):
    if pred_hist.ndim != 2 or true_hist.ndim != 2:
        raise ValueError(
            f"Both histograms must be 2D but got {pred_hist.ndim}D and {true_hist.ndim}D"
        )

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9, 4))
    visualize_histogram_2d(true_hist, vis_mode=vis_mode, ax=ax0, title="true")
    visualize_histogram_2d(pred_hist, vis_mode=vis_mode, ax=ax1, title="pred")

    pdf_buffer = matplotlib_to_pdf_buffer(fig)

    if show:
        plt.show()
    else:
        plt.close()

    return pdf_buffer


if __name__ == "__main__":
    # ------------------------------------------------------------
    # 1D example
    # ------------------------------------------------------------
    np.random.seed(0)

    data_1d = np.random.normal(loc=0.0, scale=1.0, size=10_000)
    data_1d = np.expand_dims(data_1d, -1)

    hist_1d = Histogram.from_samples(
        data_1d,
        bins=50,
    )

    print("1D histogram:")
    print(hist_1d)

    print("mean:", hist_1d.get_mean())
    print("std:", hist_1d.get_std())
    print("support:", hist_1d.get_support_range())

    # Optional visualization
    visualize_histogram_1d(
        hist_1d,
        show=True,
        title="1D Histogram",
        xlabel="x",
        ylabel="density",
        vis_mode=plot_as_density,
    )

    # ------------------------------------------------------------
    # 2D example
    # ------------------------------------------------------------
    np.random.seed(1)

    mean = [0.0, 2.0]
    cov = [[1.0, 0.5], [0.5, 2.0]]

    data_2d = np.random.multivariate_normal(
        mean,
        cov,
        size=20_000,
    )

    hist_2d = Histogram.from_samples(
        data_2d,
        bins=(60, 40),
    )

    print("\n2D histogram:")
    print(hist_2d)

    print("mean:", hist_2d.get_mean())
    print("std:", hist_2d.get_std())
    print("support:", hist_2d.get_support_range())

    # Optional visualization
    visualize_histogram_2d(
        hist_2d,
        show=True,
        title="2D Histogram",
        xlabel="x",
        ylabel="y",
        vis_mode=plot_as_log_density,
    )
