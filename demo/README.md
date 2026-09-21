# boltzeval demos

Two runnable end-to-end examples on a two-dimensional Gaussian mixture with two
modes ([gmm.py](gmm.py)). The "model" under evaluation is faked throughout by
adding a little Gaussian noise to reference data, so the demos run in seconds
and still produce a visible metric gap.

```bash
python -m demo.gmm_demo                          # run all three demos
python -m demo.gmm_demo --demo samples           # only the sample-based demo
python -m demo.gmm_demo --demo trajectories      # only the trajectory demo
python -m demo.gmm_demo --demo single            # only the single-dataset demo
python -m demo.gmm_demo --out-dir out            # also write the plots as PDFs
python -m demo.gmm_demo --show                   # also show the plots on screen
```

`--out-dir` writes every visualization as a PDF (the metric groups become
sub-directories), `--show` opens each of them in a window; the demo functions
take the same options as the `out_dir` and `show` arguments.

## `run_sample_demo` - evaluating the equilibrium distribution

Reference data are exact i.i.d. samples of the mixture, the model samples are
an independent set blurred with Gaussian noise. The pipeline is built from:

| node | what it reports |
| --- | --- |
| `SamplePlot2DNode` | scatter of both sample sets over the target log density |
| `CoordinateMarginalNode` | 1D and 2D coordinate marginals plus JSD / total variation |
| `EnergyHistNode` | histogram of the target energies and its JSD |

## `run_trajectory_demo` - evaluating the dynamics

Reference data are overdamped Langevin trajectories,

```
x_{t+1} = x_t + dt * grad log p(x_t) + sqrt(2 dt) * xi
```

whose stationary distribution is the mixture itself, so the slow process is the
hopping between the two modes.

A trajectory is not just a batch of samples: its frames are separated by a fixed
amount of physical time. The demo therefore uses `TrajectoryEnsemble` and saves
the reference at every step (`frame_stride = 0.005 ps`) and the model at every
second step (`frame_stride = 0.010 ps`), which is the normal situation when a
model emits frames more coarsely than the reference simulation.

Dynamical nodes are consequently parameterized by a physical **lag time**
(`lag_time = 0.1 ps`), never by a number of frames. Each ensemble converts that
lag into its own frame lag (20 frames for the reference, 10 for the model), and
the conversion fails loudly if the lag time is not an integer multiple of a
frame stride - comparing VAMP scores at different physical lags would be
meaningless.

| node | what it reports |
| --- | --- |
| `VampNode` | VAMP-2 score of both ensembles and their gap |
| `ImpliedTimescaleNode` | slowest relaxation timescales in ps, their relative error, and a parity plot against the reference |
| `TicaHistNode` | the sampled distribution in TICA coordinates (TICA fitted on the reference) |
| `CoordinateMarginalNode` | plain coordinate marginals of the pooled frames |

The static nodes consume the very same data as a plain sample batch, which
`TrajectoryEnsemble.as_samples()` pools for them.

## `run_single_demo` - describing one dataset on its own

Sometimes there is nothing to compare against. The `...NodeSingle` variants
require only the `*_true` fields, draw no "true"/"pred" titles or legends, and
omit every metric that needs two datasets. Their default output is the plot;
the raw data behind it is opt-in.

This demo runs **all seven** of them. One set of Langevin trajectories is the
single dataset, looked at from every angle:

| node | what it reports |
| --- | --- |
| `SamplePlot2DNodeSingle` | scatter of the frames over the target log density |
| `CoordinateMarginalNodeSingle` | 1D and 2D coordinate marginals (`include_histograms=True`) |
| `EnergyHistNodeSingle` | histogram of the target energies (`include_histogram=True`) |
| `TicaHistNodeSingle` | the distribution in TICA coordinates (`include_projections=True`) |
| `VampNodeSingle` | VAMP-2 score of the trajectories |
| `ImpliedTimescaleNodeSingle` | timescale spectrum, labelled with its values |

The seventh, `TorsionMarginalNodeSingle`, reads backbone angles off a molecule,
which the two-dimensional toy system does not have. It is therefore shown on
alanine dipeptide (`test_files/aldp_topology.pdb`), with structures faked as
noise around the reference conformation. That conformation is fully extended
(phi = psi = pi), so the density appears at the edges of the Ramachandran plot
rather than in its middle.
