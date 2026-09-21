"""
End-to-end boltzeval demos on a two-mode Gaussian mixture.

Three independent demos show the kinds of evaluation:

* :func:`run_sample_demo` evaluates i.i.d. *samples*, i.e. how well a model
  reproduces the equilibrium (Boltzmann) distribution.
* :func:`run_trajectory_demo` evaluates *trajectories*, i.e. how well a model
  reproduces the dynamics on top of that distribution.
* :func:`run_single_demo` describes *one* dataset on its own, without anything
  to compare it against, using every single-dataset node there is.

All of them build an evaluation pipeline out of nodes, hand it an ``EvalData``
container and let ``run_eval`` produce a flat dictionary of metrics.

Run both demos with::

    python -m demo.gmm_demo               # from the repository root
    python -m demo.gmm_demo --out-dir out # ... and write the plots as PDFs
    python -m demo.gmm_demo --show        # ... and show the plots on screen
"""

import argparse
import os
from pathlib import Path

import mdtraj as md
import numpy as np

from boltzeval.metrics.feature_transforms import IdentityFeatureTransform
from boltzeval.metrics.hist_comparison import (
    get_hist_jensen_shannon,
    get_hist_total_variation,
)
from boltzeval.metrics.tica import fit_tica
from boltzeval.pipeline import EvalData, get_pdfs, get_scalar_metrics, run_eval
from boltzeval.pipeline.nodes import (
    CoordinateMarginalNode,
    CoordinateMarginalNodeSingle,
    EnergyHistNode,
    EnergyHistNodeSingle,
    ImpliedTimescaleNode,
    ImpliedTimescaleNodeSingle,
    SamplePlot2DNode,
    SamplePlot2DNodeSingle,
    TicaHistNode,
    TicaHistNodeSingle,
    TorsionMarginalNodeSingle,
    VampNode,
    VampNodeSingle,
)
from boltzeval.utils.pdf import plot_pdf, save_pdfs

try:
    from demo.gmm import (
        add_gaussian_noise,
        add_gaussian_noise_to_trajectories,
        make_double_well_mixture,
        simulate_overdamped_langevin,
    )
except ImportError:  # when the script is executed from inside demo/
    from gmm import (
        add_gaussian_noise,
        add_gaussian_noise_to_trajectories,
        make_double_well_mixture,
        simulate_overdamped_langevin,
    )


def run_sample_demo(
    out_dir: str | None = None, seed: int = 0, show: bool = False
) -> dict:
    """
    Evaluate i.i.d. samples of a faked model against reference samples.

    The pipeline covers the metrics one usually wants for a Boltzmann
    generator: a scatter plot over the target density, the coordinate
    marginals, and the energy histogram.

    Parameters
    ----------
    out_dir : str | None
        If given, all PDF visualizations are written into this directory.
    seed : int
        Seed of the random number generator.
    show : bool
        Whether to display the visualizations on screen (one window per plot).

    Returns
    -------
    dict
        The flat metrics dictionary returned by ``run_eval``.
    """
    print("\n" + "=" * 70)
    print("Sample-based demo: two-mode Gaussian mixture")
    print("=" * 70)

    rng = np.random.default_rng(seed)
    target = make_double_well_mixture()

    n_samples = 20_000

    # === Reference ("true") and model ("pred") samples ===
    # The model is faked: an independent set of exact samples blurred with a
    # little Gaussian noise, which makes it slightly too broad.
    samples_true = target.sample(n_samples, rng)
    samples_pred = add_gaussian_noise(target.sample(n_samples, rng), 0.3, rng)

    # === Construct the evaluation pipeline ===
    pipeline = [
        # Scatter of both sample sets on top of the target log density
        SamplePlot2DNode(
            target_log_prob_fn=target.log_prob,
            xlim=(-4.0, 4.0),
            ylim=(-2.5, 2.5),
            log_prob_range=(-12.0, 0.0),
        ),
        # 1D marginals of x and y plus the joint 2D marginal
        CoordinateMarginalNode(
            marginals=[(0,), (1,), (0, 1)],
            hist_metrics=[get_hist_jensen_shannon, get_hist_total_variation],
            n_bins=60,
        ),
        # Histogram of the target energies -log p(x) of both sample sets
        EnergyHistNode(hist_metrics=[get_hist_jensen_shannon]),
    ]

    # === Prepare the data for evaluation ===
    # The energy histogram needs the target log probabilities of both sample sets.
    data = EvalData(
        samples_true=samples_true,
        samples_pred=samples_pred,
        true_samples_target_log_prob=target.log_prob(samples_true),
        pred_samples_target_log_prob=target.log_prob(samples_pred),
    )

    metrics = run_eval(data, pipeline=pipeline)

    _report(metrics, out_dir, show)
    return metrics


def run_trajectory_demo(
    out_dir: str | None = None, seed: int = 0, show: bool = False
) -> dict:
    """
    Evaluate trajectories of a faked model against reference trajectories.

    Trajectories carry a ``frame_stride``: the physical time between two
    consecutive frames. This demo deliberately saves the reference and the
    model trajectories at *different* strides, which is the normal situation
    when a model emits frames at a coarser resolution than the reference
    simulation. Dynamical nodes are therefore parameterized with a physical
    lag time, and each ensemble converts it into its own frame lag.

    Parameters
    ----------
    out_dir : str | None
        If given, all PDF visualizations are written into this directory.
    seed : int
        Seed of the random number generator.
    show : bool
        Whether to display the visualizations on screen (one window per plot).

    Returns
    -------
    dict
        The flat metrics dictionary returned by ``run_eval``.
    """
    print("\n" + "=" * 70)
    print("Trajectory-based demo: overdamped Langevin on a two-mode mixture")
    print("=" * 70)

    rng = np.random.default_rng(seed)
    target = make_double_well_mixture()

    # === Reference trajectories: overdamped Langevin, saved every step ===
    trajs_true = simulate_overdamped_langevin(
        target,
        rng,
        n_trajectories=8,
        n_steps=20_000,
        time_step=0.005,  # ps
        save_stride=1,  # -> frame_stride = 0.005 ps
    )

    # === Model trajectories: faked by simulating at a coarser save stride
    #     and blurring the frames with a little Gaussian noise ===
    trajs_pred = simulate_overdamped_langevin(
        target,
        rng,
        n_trajectories=8,
        n_steps=20_000,
        time_step=0.005,  # ps
        save_stride=2,  # -> frame_stride = 0.010 ps
    )
    trajs_pred = add_gaussian_noise_to_trajectories(trajs_pred, 0.3, rng)

    print(f"reference trajectories: {trajs_true}")
    print(f"model trajectories:     {trajs_pred}")

    # A physical lag time, not a number of frames: it is an integer multiple of
    # both frame strides (20 frames of trajs_true, 10 frames of trajs_pred).
    lag_time = 0.1  # ps
    feature_transform = IdentityFeatureTransform()

    # TICA is fitted on the reference trajectories only and then used as a
    # fixed projection for both sample sets.
    tica_model = fit_tica(
        trajectories=trajs_true,
        lag_time=lag_time,
        feature_transform=feature_transform,
        dim=2,
    )

    # === Construct the evaluation pipeline ===
    pipeline = [
        # Dynamical: how much slow dynamics does the model capture?
        VampNode(lag_time=lag_time, feature_transform=feature_transform, r=2.0),
        # Dynamical: relaxation timescales, reported in ps, plus a parity plot.
        # Without n_timescales, all available ones are compared (here: two,
        # one per feature dimension of the identity transform).
        ImpliedTimescaleNode(lag_time=lag_time, feature_transform=feature_transform),
        # Static: the distribution the trajectories sample, in TICA coordinates
        TicaHistNode(
            tica=tica_model,
            feature_transform=feature_transform,
            hist_metrics=[get_hist_jensen_shannon],
        ),
        # Static: the plain coordinate marginals
        CoordinateMarginalNode(
            marginals=[(0,), (1,)],
            hist_metrics=[get_hist_jensen_shannon],
            n_bins=60,
        ),
    ]

    # === Prepare the data for evaluation ===
    # Trajectory nodes consume the ensembles; the static nodes consume the same
    # data as a plain sample batch, which as_samples() pools for them.
    data = EvalData(
        trajs_true=trajs_true,
        trajs_pred=trajs_pred,
        samples_true=trajs_true.as_samples(),
        samples_pred=trajs_pred.as_samples(),
    )

    metrics = run_eval(data, pipeline=pipeline)

    _report(metrics, out_dir, show)
    return metrics


def run_single_demo(
    out_dir: str | None = None, seed: int = 0, show: bool = False
) -> dict:
    """
    Describe a single dataset with every single-dataset node there is.

    Sometimes there is nothing to compare against: a reference simulation has
    just finished, or a model was sampled without a ground truth at hand. The
    ``...NodeSingle`` variants cover that case. They require only the ``*_true``
    fields, produce visualizations without "true"/"pred" titles or legends, and
    omit every metric that needs two datasets. Their raw data (histograms, TICA
    projections, timescales) is opt-in.

    One set of Langevin trajectories serves as that single dataset here, and is
    looked at from every angle: as a scatter over the target density, as
    coordinate marginals, as an energy histogram, in TICA coordinates, and
    through its dynamics. Only the torsion marginals need a molecular system,
    so they are shown on alanine dipeptide instead.

    Parameters
    ----------
    out_dir : str | None
        If given, all PDF visualizations are written into this directory.
    seed : int
        Seed of the random number generator.
    show : bool
        Whether to display the visualizations on screen (one window per plot).

    Returns
    -------
    dict
        The flat metrics dictionary returned by ``run_eval``.
    """
    print("\n" + "=" * 70)
    print("Single-dataset demo: describing one dataset on its own")
    print("=" * 70)

    rng = np.random.default_rng(seed)
    target = make_double_well_mixture()

    # === The one dataset: overdamped Langevin trajectories ===
    trajs = simulate_overdamped_langevin(
        target,
        rng,
        n_trajectories=8,
        n_steps=10_000,
        time_step=0.005,  # ps
        save_stride=1,  # -> frame_stride = 0.005 ps
    )
    # The static nodes want the same data as a plain sample batch. The pooled
    # frames are in time order and strongly correlated, so shuffle them: nodes
    # that subsample the batch (the scatter plot) would otherwise only ever see
    # the beginning of the first trajectory.
    samples = trajs.as_samples()
    rng.shuffle(samples)

    print(f"trajectories: {trajs}")
    print(f"pooled frames: {samples.shape}")

    lag_time = 0.1  # ps
    feature_transform = IdentityFeatureTransform()

    tica_model = fit_tica(
        trajectories=trajs,
        lag_time=lag_time,
        feature_transform=feature_transform,
        dim=2,
    )

    # === Construct the evaluation pipeline ===
    pipeline = [
        # Where the samples lie on the target density
        SamplePlot2DNodeSingle(
            target_log_prob_fn=target.log_prob,
            xlim=(-4.0, 4.0),
            ylim=(-2.5, 2.5),
            log_prob_range=(-12.0, 0.0),
        ),
        # The coordinate marginals, keeping the raw histograms
        CoordinateMarginalNodeSingle(
            marginals=[(0,), (1,), (0, 1)],
            n_bins=60,
            include_histograms=True,
        ),
        # The distribution of target energies
        EnergyHistNodeSingle(include_histogram=True),
        # The distribution in TICA coordinates, keeping the raw projections
        TicaHistNodeSingle(
            tica=tica_model,
            feature_transform=feature_transform,
            include_projections=True,
        ),
        # How much slow dynamics there is to capture
        VampNodeSingle(lag_time=lag_time, feature_transform=feature_transform),
        # The relaxation timescales, as a labelled spectrum
        ImpliedTimescaleNodeSingle(
            lag_time=lag_time,
            feature_transform=feature_transform,
            include_timescales=True,
            annotate_timescales=True,
        ),
    ]

    # === Prepare the data for evaluation ===
    # Note that only the *_true fields are needed: there is no prediction here.
    data = EvalData(
        samples_true=samples,
        true_samples_target_log_prob=target.log_prob(samples),
        trajs_true=trajs,
    )

    metrics = run_eval(data, pipeline=pipeline)

    # The torsion marginals are the one single node the 2D toy system cannot
    # show, since they need a molecule to read backbone angles off.
    metrics.update(_run_torsion_single(rng))

    _report(metrics, out_dir, show)
    return metrics


def _run_torsion_single(rng: np.random.Generator) -> dict:
    """
    Run TorsionMarginalNodeSingle on alanine dipeptide.

    The structures are faked the same way as everywhere else in these demos, by
    adding Gaussian noise to the atom positions of the reference conformation.
    That conformation is fully extended (phi = psi = pi), so the resulting
    density sits at the periodic boundary and shows up at the edges of the
    Ramachandran plot rather than in its middle.
    """
    topology_path = Path(__file__).resolve().parent / "test_files"
    topology_path = topology_path / "aldp_topology.pdb"

    if not topology_path.is_file():
        print(f"\nSkipping the torsion marginals: {topology_path} not found.")
        return {}

    pdb = md.load(str(topology_path))
    print(f"\nalanine dipeptide: {pdb.topology}")

    # (n_samples, n_atoms, 3); EvalData flattens this to (n_samples, n_atoms * 3)
    samples = pdb.xyz[0] + 0.02 * rng.normal(size=(2_000, *pdb.xyz[0].shape))

    data = EvalData(samples_true=samples)
    pipeline = [
        TorsionMarginalNodeSingle(
            topology=pdb.topology,
            include_free_energy_difference=True,
        )
    ]
    return run_eval(data, pipeline=pipeline)


def _report(metrics: dict, out_dir: str | None, show: bool = False):
    """
    Print the scalar metrics, and write and/or display the visualizations.
    """
    scalars = get_scalar_metrics(metrics)
    print("\n--- scalar metrics ---")
    for key, value in sorted(scalars.items()):
        print(f"{key:<55s} {value: .6f}")

    pdfs = get_pdfs(metrics)

    # Whatever is neither a scalar nor a plot is raw data a node was asked to
    # hand back, e.g. histograms or TICA projections
    raw_data = {
        k: v for k, v in metrics.items() if k not in scalars if k not in pdfs
    }
    if raw_data:
        print(f"\n--- {len(raw_data)} raw data entries ---")
        for key, value in sorted(raw_data.items()):
            if isinstance(value, np.ndarray):
                value = f"ndarray(shape={value.shape})"
            print(f"{key:<55s} {value}")

    print(f"\n--- {len(pdfs)} visualization(s) ---")

    if out_dir is None:
        for key, pdf in sorted(pdfs.items()):
            print(f"{key:<55s} {pdf}")
    else:
        # Metric keys are grouped with "/", which becomes a sub-directory here
        fpaths = save_pdfs(pdfs, out_dir)
        for key, fpath in sorted(fpaths.items()):
            print(f"{key:<55s} -> {fpath}")

    if show:
        # One blocking window per visualization
        for key, pdf in sorted(pdfs.items()):
            plot_pdf(pdf, title=key, show=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory the PDF visualizations are written to (default: none)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display the visualizations on screen (one window per plot)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--demo",
        choices=["samples", "trajectories", "single", "all"],
        default="all",
        help="which demo to run (default: all)",
    )
    args = parser.parse_args()

    def sub_dir(name: str) -> str | None:
        return None if args.out_dir is None else os.path.join(args.out_dir, name)

    if args.demo in ("samples", "all"):
        run_sample_demo(out_dir=sub_dir("samples"), seed=args.seed, show=args.show)

    if args.demo in ("trajectories", "all"):
        run_trajectory_demo(
            out_dir=sub_dir("trajectories"), seed=args.seed, show=args.show
        )

    if args.demo in ("single", "all"):
        run_single_demo(out_dir=sub_dir("single"), seed=args.seed, show=args.show)


if __name__ == "__main__":
    main()
