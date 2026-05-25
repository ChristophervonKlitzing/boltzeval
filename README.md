# boltzeval (Boltzmann evaluation)
Implementation of common metrics, visualizations, and a modular evaluation pipeline for molecular Boltzmann tasks. 


## Metrics
Metrics for manual use are implemented under `boltzeval.metrics`.

## Building and running an evaluation pipeline
Evaluation is performed by composing a list of modular evaluation nodes.

### Example
```python
from boltzeval.pipeline import run_eval, EvalData
from boltzeval.pipeline.energy_hist import EnergyHistComparison
from boltzeval.metrics.hist_comparison import get_hist_jensen_shannon

# === Construct evaluation pipeline ===
eval_pipeline = []
eval_pipeline.append(EnergyHistEval(hist_metrics=[get_hist_jensen_shannon]))

# === Prepare data for evaluation ===
data = EvalData(true_samples_target_log_prob=..., pred_samples_target_log_prob=...)

metrics = run_eval(data, pipeline=eval_pipeline)
```

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

TODO

#### Histograms
The density-counts of histograms can be exported for custom downstream analysis (e.g., visualizations):

TODO


