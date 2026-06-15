from typing import Callable

from matplotlib import pyplot as plt
import numpy as np

from boltzeval.utils.pdf import matplotlib_to_pdf_buffer


def plot_2d(
    log_prob_fn: Callable[[np.ndarray], np.ndarray],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    ax: plt.Axes | None = None,
    resolution=300,
    log_prob_range: tuple[float, float] | None = None,
    cmap="RdYlGn_r",
    show: bool = False,
):
    new_plot = ax is None
    if new_plot:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    XX, YY = np.meshgrid(xs, ys)

    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)

    ZZ = log_prob_fn(grid)
    ZZ = ZZ.reshape(resolution, resolution)

    if log_prob_range is not None:
        vmin, vmax = log_prob_range
        ZZ = np.clip(ZZ, vmin, vmax)

    cf = ax.contourf(XX, YY, ZZ, levels=60, cmap=cmap, alpha=0.85)
    ax.contour(XX, YY, ZZ, levels=20, colors="k", linewidths=0.3, alpha=0.4)

    fig.colorbar(cf, ax=ax, label="log probability")

    pdf_buffer = None

    if show:
        plt.show()
    elif new_plot:
        pdf_buffer = matplotlib_to_pdf_buffer(fig)
        plt.close(fig)

    return pdf_buffer
