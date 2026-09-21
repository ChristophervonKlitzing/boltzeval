# boltzeval (Boltzmann evaluation)
Implementation of common metrics, visualizations, and a modular evaluation pipeline for molecular Boltzmann tasks. 


## Metrics
Metrics for manual use are implemented under `boltzeval.metrics`.

## Demos
Two runnable end-to-end examples on a two-mode Gaussian mixture live in
[`demo/`](demo/): one for sample-based evaluation and one for trajectory-based
evaluation.

```bash
python -m demo.gmm_demo
```

## Building and running an evaluation pipeline
Evaluation is performed by composing a list of modular evaluation nodes.

### Example
```python
from boltzeval.pipeline import run_eval, EvalData
from boltzeval.pipeline.nodes import EnergyHistNode
from boltzeval.metrics.hist_comparison import get_hist_jensen_shannon

# === Construct evaluation pipeline ===
eval_pipeline = []
eval_pipeline.append(EnergyHistNode(hist_metrics=[get_hist_jensen_shannon]))

# === Prepare data for evaluation ===
data = EvalData(true_samples_target_log_prob=..., pred_samples_target_log_prob=...)

metrics = run_eval(data, pipeline=eval_pipeline)
```

### Single-dataset nodes
Every node comes in two flavors:

- `<Name>Node` compares a prediction against a reference and therefore requires
  both, e.g. `samples_true` **and** `samples_pred`.
- `<Name>NodeSingle` looks at one dataset on its own and requires only the
  `*_true` fields.

The single variants exist for visualizing a dataset rather than scoring a model
against it: their plots carry no "true"/"pred" titles or legends, and metrics
that need both datasets (histogram comparisons, score gaps, relative errors)
are omitted. Their default output is the visualization; the underlying raw data
is opt-in:

```python
from boltzeval.pipeline.nodes import TicaHistNodeSingle

eval_pipeline.append(
    TicaHistNodeSingle(
        tica=tica_model,
        feature_transform=feature_transform,
        include_histogram=True,    # off by default
        include_projections=True,  # off by default
    )
)

data = EvalData(samples_true=samples)
```

Available pairs: `EnergyHistNode`, `TicaHistNode`, `CoordinateMarginalNode`,
`SamplePlot2DNode`, `TorsionMarginalNode`, `VampNode` and
`ImpliedTimescaleNode`, each with its `...Single` counterpart.

### Trajectories
Samples are plain arrays, but a trajectory is more than a batch of samples: its
frames are separated by a fixed amount of physical time. That time cannot be
recovered from an array, and without it dynamical metrics are silently
incomparable - the same number of frames means a different physical lag for
trajectories saved at different strides.

Trajectory fields of `EvalData` therefore take a `TrajectoryEnsemble`: a set of
`Trajectory` objects that bundle the frames with the `frame_stride` (the
physical time between two consecutive frames) and an optional `time_unit` label.

```python
from boltzeval.pipeline import EvalData, TrajectoryEnsemble

# frames: (n_trajectories, n_frames, dim) or a list of (n_frames, dim) arrays
trajs_true = TrajectoryEnsemble.from_array(frames, frame_stride=0.005, time_unit="ps")
trajs_pred = TrajectoryEnsemble.from_array(model_frames, frame_stride=0.01, time_unit="ps")

data = EvalData(
    trajs_true=trajs_true,
    trajs_pred=trajs_pred,
    # the same frames pooled into a plain sample batch, for the static nodes
    samples_true=trajs_true.as_samples(),
    samples_pred=trajs_pred.as_samples(),
)
```

Reference and predicted trajectories may use different frame strides, but they
must use the same time unit - no unit conversion is ever performed, and keeping
units consistent is up to the caller.

Dynamical nodes (`VampNode`, `ImpliedTimescaleNode`) and `fit_tica` are
parameterized by a physical `lag_time` rather than a number of frames. Every
ensemble converts that lag into its own frame lag, and the conversion raises if
the lag time is not an integer multiple of the frame stride, or does not fit
into the trajectories:

```python
from boltzeval.pipeline.nodes import VampNode

# 0.1 ps = 20 frames of trajs_true = 10 frames of trajs_pred
eval_pipeline.append(VampNode(lag_time=0.1, feature_transform=feature_transform))
```

Frames of shape `(n_frames, n_atoms, 3)` are flattened to
`(n_frames, n_atoms * 3)` automatically, and trajectories of differing length
are supported by passing a list of arrays.

### Missing data behavior
By default, evaluation modules that cannot be computed due to missing fields in EvalData are skipped with a warning.

To enforce strict execution (raise an error instead), disable skipping:

```python
metrics = run_eval(data, skip_on_missing_data=False)
```


### Output Format

The returned `metrics` object is a flat dictionary:

- **Keys**: metric names (strings)
- **Values** can be:
  - Scalars (float, int)
  - Structured data (e.g., histograms)
  - Binary artifacts (e.g., PDF files stored as byte buffers)

Utility functions are available to extract or filter subsets of metrics depending on downstream use.


### Logging and Export

#### Weights & Biases Logging

wandb is optional and must be installed separately.

To log evaluation results:

```python
from boltzeval.pipeline import make_wandb_compatible
import wandb

# transforms all metrics into wandb-compatible ones
# e.g., converts pdfs into low-resolution images.
# Drops raw data like histograms to not clutter the wandb server.
wandb_metrics = make_wandb_compatible(metrics)
wandb.log(wandb_metrics)
```

#### PDF artifacts
Some evaluations produce visualizations as PDF files stored in-memory as binary buffers. These can be written directly to disk:
```python
from boltzeval.pipeline import get_pdfs
from boltzeval.utils.pdf import save_pdfs

pdfs = get_pdfs(metrics)
save_pdfs(pdfs, "out/pdfs")  # saved as .pdf
```


#### Histograms
The density-counts of histograms can be exported for custom downstream analysis (e.g., visualizations):
```python
from boltzeval.pipeline import get_histograms
from boltzeval.utils.histogram import save_histograms

hists = get_histograms(metrics)
save_histograms(hists, "out/histograms")  # saved as .npz (bins and counts)
```


