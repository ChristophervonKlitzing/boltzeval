from matplotlib import pyplot as plt
import numpy as np
import mdtraj as md

from boltzeval.utils.hist_visualization import (
    VisualizationMode,
    plot_as_log_density,
    visualize_histogram_1d,
    visualize_histogram_2d,
)
from boltzeval.utils.histogram import Histogram
from boltzeval.utils.pdf import matplotlib_to_pdf_buffer


def get_trajectory(samples: np.ndarray, topology: md.Topology):
    """
    Get trajectory from samples and topology together with potential reshaping.
    samples can be given as (batch, #atoms, 3) or (batch, #atoms * 3).
    """
    batch = samples.shape[0]
    samples = samples.reshape(batch, -1, 3)
    assert topology.n_atoms == samples.shape[1]

    traj_samples = md.Trajectory(samples, topology=topology)
    return traj_samples


def get_torsion_angles(samples: np.ndarray, topology: md.Topology):
    """
    Extract backbone φ and ψ torsion angles in the range (-pi, pi) from a batch of molecular coordinates.

    Parameters
    ----------
    samples : np.ndarray
        Cartesian coordinates of shape (batch, n_atoms, 3) or (batch, n_atoms*3),
        where `batch` is the number of frames and the last dimension corresponds to (x, y, z).
    topology : md.Topology
        MDTraj topology describing the molecular system.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of arrays `(phis, psis)`:
        - phis : np.ndarray of shape (batch, n_torsions)
        - psis : np.ndarray of shape (batch, n_torsions)
        Each entry contains the backbone φ or ψ angles in radians.
    """
    traj_samples = get_trajectory(samples, topology)
    phis = md.compute_phi(traj_samples)[1]
    psis = md.compute_psi(traj_samples)[1]
    return phis, psis


def get_torsion_marginal_hists(
    phis: np.ndarray, psis: np.ndarray, **kwargs
) -> tuple[list[Histogram], list[Histogram], list[Histogram]]:
    """
    Compute 1D and 2D histograms of torsion angles for all φ/ψ pairs.

    Parameters
    ----------
    phis : np.ndarray
        Array of shape (batch, n_torsions) with φ angles in radians in range (-pi, pi).
    psis : np.ndarray
        Array of shape (batch, n_torsions) with ψ angles in radians in range (-pi, pi).
    **kwargs
        Additional keyword arguments forwarded to `get_histogram_1d` and `get_histogram_2d`.

    Returns
    -------
    tuple[Histogram2D, Histogram1D, Histogram1D]
        A tuple of equal-size lists (one list element per torsion angle pair).
        - "phi_psi": 2D `Histogram2D` of the joint φ/ψ angles
        - "phi": 1D `Histogram1D` of φ angles
        - "psi": 1D `Histogram1D` of ψ angles
    """
    angles = np.stack([phis, psis], -1)  # (batch, <num-angle-pairs>, 2)

    range_1d = (-np.pi, np.pi)
    range_2d = ((-np.pi, np.pi), (-np.pi, np.pi))

    phi_psi_hists = []
    phi_hists = []
    psi_hists = []

    for angle_pair_idx in range(angles.shape[1]):
        # 2D histogram
        hist_2d = Histogram.from_samples(
            angles[:, angle_pair_idx, :], data_range=range_2d, **kwargs
        )
        phi_psi_hists.append(hist_2d)

        # 2 1D histograms
        phi_hist_1d = Histogram.from_samples(
            angles[:, angle_pair_idx, 0], data_range=range_1d, **kwargs
        )
        phi_hists.append(phi_hist_1d)

        psi_hist_1d = Histogram.from_samples(
            angles[:, angle_pair_idx, 1], data_range=range_1d, **kwargs
        )
        psi_hists.append(psi_hist_1d)

    return phi_psi_hists, phi_hists, psi_hists


def get_free_energy_difference(
    phis: np.ndarray,
    phi_range: tuple[float, float] = (0, 2),
):
    """
    Estimate the free energy difference between two regions of backbone torsion phi angles.

    The function partitions the input values into an "inner" region defined
    by `phi_range` and an "outer" region (everything outside that interval),
    then computes a free energy difference based on the relative populations
    of these regions:

        ΔF = - (log(N_inner) - log(N_outer)) = log(N_outer / N_inner)

    Parameters
    ----------
    phis : np.ndarray
        Array of sampled values. Can be of shape (batch,) or
        (batch, n_torsions). All values are flattened implicitly when counting.
    phi_range : tuple of float, optional
        Tuple (phi_min, phi_max) defining the inclusive bounds of the
        "inner" region. Default is (0, 2).

    Returns
    -------
    float
        Estimated free energy difference ΔF in $kB T$ (i.e., no multiplication with kB*T)
        between the outer and inner regions. Returns np.nan if either region contains zero samples.
    """
    phi_min, phi_max = phi_range
    inner_mask = (phis >= phi_min) & (phis <= phi_max)
    outer_mask = (phis < phi_min) | (phis > phi_max)

    inner_count = np.sum(inner_mask)
    outer_count = np.sum(outer_mask)

    if inner_count > 0 and outer_count > 0:
        free_energy_delta = -(np.log(inner_count) - np.log(outer_count))
    else:
        free_energy_delta = np.nan

    return free_energy_delta


def visualize_torsion_marginals_dual(
    torsion_marginals_true: tuple[list[Histogram], list[Histogram], list[Histogram]],
    torsion_marginals_pred: tuple[list[Histogram], list[Histogram], list[Histogram]],
    vis_mode: VisualizationMode = plot_as_log_density,
    show: bool = False,
    cmap: str | None = None,
    **kwargs,
):
    """
    Compare true and predicted torsion-angle histograms.

    Creates side-by-side visualizations of Ramachandran (φ/ψ) 2D histograms
    and marginal 1D φ and ψ histograms for each torsion pair, and returns
    the resulting figure as a PDF buffer.

    Parameters
    ----------
    torsion_marginals_true : tuple[list[Histogram], list[Histogram], list[Histogram]]
        Ground-truth histograms in order (phi_psi_2d, phi_1d, psi_1d).

    torsion_marginals_pred : tuple[list[Histogram], list[Histogram], list[Histogram]]
        Predicted histograms in the same structure as `torsion_marginals_true`.

    vis_mode : VisualizationMode, default=plot_as_log_density
        Visualization mode passed to histogram plotting functions.

    show : bool, default=False
        If True, displays the figure interactively.

    cmap : str | None, optional
        Colormap for 2D histogram visualization.

    **kwargs
        Additional keyword arguments forwarded to histogram visualization.

    Returns
    -------
    bytes
        PDF buffer of the generated Matplotlib figure.
    """
    assert len(torsion_marginals_true[0]) == len(torsion_marginals_true[1])
    assert len(torsion_marginals_true[1]) == len(torsion_marginals_true[2])

    assert len(torsion_marginals_pred[0]) == len(torsion_marginals_pred[1])
    assert len(torsion_marginals_pred[1]) == len(torsion_marginals_pred[2])

    n_pairs = len(torsion_marginals_true[0])

    fig, axes = plt.subplots(n_pairs, 4, squeeze=False, figsize=(13, 3 * n_pairs))
    for i in range(n_pairs):
        ax_ram_true: plt.Axes = axes[i, 2]
        ax_ram_pred: plt.Axes = axes[i, 3]
        ax_phi: plt.Axes = axes[i, 0]
        ax_psi: plt.Axes = axes[i, 1]

        h_ram_true = torsion_marginals_true[0][i]
        h_phi_true = torsion_marginals_true[1][i]
        h_psi_true = torsion_marginals_true[2][i]

        h_phi_pred = torsion_marginals_pred[1][i]
        h_ram_pred = torsion_marginals_pred[0][i]
        h_psi_pred = torsion_marginals_pred[2][i]

        phi_label = f"$\\phi_{i}$"
        psi_label = f"$\\psi_{i}$"

        visualize_histogram_2d(
            h_ram_true,
            vis_mode=vis_mode,
            ax=ax_ram_true,
            title="True",
            xlabel=phi_label,
            ylabel=psi_label,
            cmap=cmap,
            **kwargs,
        )

        visualize_histogram_2d(
            h_ram_pred,
            vis_mode=vis_mode,
            ax=ax_ram_pred,
            title="Pred",
            xlabel=phi_label,
            ylabel=psi_label,
            cmap=cmap,
            **kwargs,
        )

        visualize_histogram_1d(
            h_phi_true,
            vis_mode=vis_mode,
            ax=ax_phi,
            label="True",
            xlabel=phi_label,
            **kwargs,
        )
        visualize_histogram_1d(
            h_phi_pred,
            vis_mode=vis_mode,
            ax=ax_phi,
            label="Pred",
            xlabel=phi_label,
            **kwargs,
        )

        visualize_histogram_1d(
            h_psi_true,
            vis_mode=vis_mode,
            ax=ax_psi,
            label="True",
            xlabel=psi_label,
            **kwargs,
            transpose=False,
        )
        visualize_histogram_1d(
            h_psi_pred,
            vis_mode=vis_mode,
            ax=ax_psi,
            label="Pred",
            xlabel=psi_label,
            **kwargs,
            transpose=False,
        )

    pdf = matplotlib_to_pdf_buffer(fig)

    if show:
        plt.show()
    else:
        plt.close()

    return pdf
