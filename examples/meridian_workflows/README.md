# Meridian notebook workflows as a Python project

[English](README.md) | [简体中文](README.zh-CN.md)

This is a learning project, not a new modeling library. It converts **all nine
notebooks in `demo/`** into named Python functions and independent CLI stages.
It calls the same Meridian APIs without Jupyter, Colab, notebook magics, or
hidden cell state. Original notebooks and library code are unchanged.

## 1. Where to run it

**For your Windows machine with an NVIDIA GPU: keep using WSL Ubuntu.**
WSL is a Linux execution environment, not a notebook requirement. A `.py`
program in a VS Code WSL terminal uses the same Python environment and GPU as
the notebook did.

| Environment | Recommended execution |
| --- | --- |
| WSL Ubuntu + NVIDIA CUDA | `--device cuda` |
| Native Linux + NVIDIA CUDA | `--device cuda` |
| Linux/WSL without a GPU | `--device cpu` (the safe default) |
| Native Windows or macOS | CPU may work if all Meridian dependencies install; not certified here |

The project deliberately rejects `--device cuda` on native Windows. Use a
WSL terminal, rather than PowerShell's Windows Python, for your working GPU
setup. Native Windows CPU support depends on the installed Meridian/JAX/
TensorFlow versions. You do **not** need Jupyter or a running notebook kernel.

### Use your existing WSL environment

From the repository root:

```bash
source ~/.venvs/meridian/bin/activate
python -m pip install -e './examples/meridian_workflows[all,dev]'
cd examples/meridian_workflows
python -m meridian_workflows list
```

The package depends on the current Meridian 2.1 API. When developing this
checkout, install Meridian itself editable rather than using an unrelated
older PyPI release:

```bash
# From the repository root; CUDA dependencies are for Linux/WSL.
python -m pip install -e '.[and-cuda,schema]'
```

For CPU-only installation use `.[schema]` instead. Optional example extras:
`calibration` (GeoX), `tracking` (MLflow), `scenario` (Scenario Planner), or `all`.
Installing extras is not necessary for the basic Getting Started workflow.
Do not run package installation commands inside your Python scripts.

## 2. Start here: one workflow, separate processes

Run these from this project directory. `outputs/` is gitignored.

```bash
# Inspect the input and model structure. No sampling.
python -m meridian_workflows getting-started prepare --device cpu

# A SHORT TEST of the pipeline, not a fit suitable for business decisions.
python -m meridian_workflows getting-started train --profile smoke --device cuda

# New Python process: reads the model saved above; never starts sampling.
python -m meridian_workflows getting-started report --device cpu
```

Open `outputs/getting-started/health_card.html` in your browser. Also inspect
`summary.html`, `rhat.html`, `model_fit.html`, and `parameter_diagnostics.csv`.
HTML charts use the usual Vega/Altair JavaScript CDN, so rendering charts may
need an Internet connection even though the computations run locally.

The `report` stage also writes **`residual_diagnostics.html`** and
**`residual_diagnostics.csv`**, adding **Durbin-Watson** and the
**Jarque-Bera statistic/p-value**. These reuse the model-fit chart's posterior
mean predictions; no additional sampling is needed. Residuals are actual minus
predicted outcome, on the same revenue scale as the chart when revenue per KPI
is available, otherwise on the KPI scale.

Each geo is evaluated separately in chronological order. For multiple geos,
an additional national row sums residuals across geos at each time point; it
does not concatenate geos into one time series. These are **All Data** checks,
including any holdout observations, not separate train/test evaluations.
Constant/all-zero or too-short residual series produce explicitly explained
undefined values rather than a false pass.

Durbin-Watson near 2 suggests little first-order autocorrelation, but no
universal pass/fail cutoff is applied. Jarque-Bera uses an asymptotic chi-square
p-value; short series, fitted residuals, heteroscedasticity, and serial
dependence limit its interpretation. This p-value is **not** Meridian's
Bayesian posterior predictive p-value. The supplementary report displays the
Meridian health status and does not change its score or convergence checks.
It is not a direct test of the Bayesian likelihood or a substitute for a
converged model.

Training saves `model.binpb` **before any optional report generation**. This
contains input data, model configuration, and inference samples. `model.json`
records the actual sampling settings, version, timing, and smoke/full label.
Closing VS Code or exiting Python does not erase these files.

Training refuses to replace an existing model unless you use `--overwrite`.
Prefer separate output directories for separate experiments:

```bash
python -m meridian_workflows getting-started train \
  --profile full --device cuda --output-dir outputs/getting-started-full

python -m meridian_workflows getting-started report \
  --output-dir outputs/getting-started-full

# Only after inspecting diagnostics and establishing adequate convergence:
python -m meridian_workflows getting-started optimize \
  --output-dir outputs/getting-started-full
```

`full` means the notebook-scale sample counts, **not** a certificate that a
model converged. Analysis/optimization commands guard against high/undefined
R-hat for nonconstant parameters and observed divergences. This guard is not
a substitute for the full health review, effective sample size, trace plots,
or model validation. `--allow-unconverged` bypasses the guard with an explicit
warning for learning/debugging only; such output must not guide spending.

### The flow you are learning

```text
CSV
  -> InputData: which columns are KPI, controls, media, spend, reach, etc.?
  -> ModelSpec + priors: what assumptions are made?
  -> Meridian: combine data and assumptions
  -> sample_prior / sample_posterior: expensive Bayesian computation
  -> model.binpb: persistent boundary between training and analysis
  -> reviewer / visualizer / summarizer: model quality and interpretation
  -> BudgetOptimizer: proposed allocations, conditional on a valid model
```

`prepare` only builds/describes the input and model. `train` is the only normal
stage that runs posterior sampling. `report`, `analyze`, `optimize`, and
`export` load saved samples. Missing models are errors, never a trigger to
silently train again.

## 3. Project map

```text
meridian_workflows/
  __main__.py                 python -m entry point
  cli.py                     command parsing and explicit demo dispatch
  config.py                  CPU/CUDA setup and sampling settings
  data.py                    common CSV loaders and model builders
  pipeline.py                training, atomic model saving, reports, guards
  residual_diagnostics.py    per-geo/national Durbin-Watson and Jarque-Bera
  full_funnel_analysis.py    standalone two-stage mathematical helpers
  demos/                     ALL notebook entry-point functions live here
    getting_started.py
    getting_started_tf.py
    reach_frequency.py
    response_curves.py
    prior_calibration.py
    full_funnel.py
    scenario_planner.py
    mlflow_demo.py
    rf_simulation.py
tests/                       focused real-data and integration checks
```

Start reading `demos/getting_started.py`, then `data.py`, `config.py`, and
`pipeline.py`. Read the full-funnel math last: it is an advanced, experimental
extension copied into a regular Python module, not a prerequisite for the
basic workflow.

Each demo exposes `run(options)`. The model-building demos also expose a
`build_model(...)` function where their specific assumptions are visible.
There is no dynamic notebook execution, `exec` of cell text, or dependence
on notebook files at runtime.

### Call a demo from your own `.py` file

```python
from pathlib import Path
from meridian_workflows.config import Options, configure_runtime

configure_runtime(backend="jax", device="cpu")

# Configure BEFORE importing Meridian or a demo that imports it.
from meridian_workflows.demos import getting_started

getting_started.run(
    Options(output=Path("outputs/my-example"), action="prepare")
)
```

To change backend/device, start a new Python process. A `run()` function is an
ordinary function, not a notebook cell with a hidden kernel.

## 4. Notebook-to-code guide and every demo

| Original notebook | CLI demo / Python module | Main lesson |
| --- | --- | --- |
| [Getting Started](../../demo/Meridian_Getting_Started.ipynb) | `getting-started` / `getting_started.py` | Data, priors, AKS, samples, health, reports, optimization and hypothetical data |
| [Getting Started TF](../../demo/Meridian_Getting_Started_Tf.ipynb) | `getting-started-tf` / `getting_started_tf.py` | Same stages with legacy TensorFlow, plus pre-modeling EDA |
| [RF Demo](../../demo/Meridian_RF_Demo.ipynb) | `reach-frequency` / `reach_frequency.py` | Impressions versus reach/frequency and RF priors |
| [ROI/mROI](../../demo/ROI_mROI_Response_Curves.ipynb) | `response-curves` / `response_curves.py` | Credible intervals, uncertainty, profitability probability, response curves |
| [Prior Calibration](../../demo/Meridian_Prior_Calibration_JAX.ipynb) | `prior-calibration` / `prior_calibration.py` | GeoX, generic experiments and combined calibrated priors |
| [Full Funnel](../../demo/Meridian_Full_Funnel.ipynb) | `full-funnel` / `full_funnel.py` | Brand advertising -> brand equity -> conversions |
| [Scenario Planner](../../demo/Meridian_Scenario_Planner_Beta.ipynb) | `scenario-planner` / `scenario_planner.py` | Optimization grids, local tables, optional Sheets/Looker integration |
| [MLflow](../../demo/Meridian_MLflow_Demo.ipynb) | `mlflow` / `mlflow_demo.py` | Reproducible local experiment tracking |
| [RF Simulation](../../demo/RF_Data_Simulation_for_Meridian.ipynb) | `rf-simulation` / `rf_simulation.py` | Generate synthetic data with known effects and ROI |

The shared `prepare`, `train`, `report`, and `optimize` actions apply to the
single-model demos. Use `python -m meridian_workflows list` for the complete
action list. The commands below are examples, not a script to run all at once.

### Getting Started: a hypothetical spending scenario

```bash
python -m meridian_workflows getting-started hypothetical \
  --output-dir outputs/getting-started-full
```

This uses the notebook's hypothetical CSV, budget of 50 million, and channel
allocation proportions. Geos and dates are explicitly aligned; missing or
duplicate geo/date records are errors instead of silently filled zeros.

### TensorFlow variant

```bash
python -m meridian_workflows getting-started-tf prepare
python -m meridian_workflows getting-started-tf eda --prior 50
python -m meridian_workflows getting-started-tf train --device cuda --profile smoke
```

Meridian's TensorFlow backend is deprecated. This entry point is retained to
explain its notebook, not to recommend it over JAX for new work.

### Reach and frequency

```bash
python -m meridian_workflows reach-frequency prepare
python -m meridian_workflows reach-frequency train --device cuda --profile smoke
python -m meridian_workflows reach-frequency report
```

There are three impression-based channels and one RF channel. Compare the
builder's `with_media()` and `with_reach()` calls and its RF ROI prior.

### ROI, marginal ROI, and response curves

```bash
# Reuse an existing fitted model instead of training another one.
python -m meridian_workflows response-curves analyze \
  --model outputs/getting-started-full/model.binpb
```

Outputs include `roi_credible_intervals.csv`, `mroi_credible_intervals.csv`,
`posterior_uncertainty.csv`, and `response_curves.html`. To reproduce this
notebook's exact input structure instead, use its own `prepare`/`train`
actions; unlike Getting Started, it excludes the organic and promo inputs
and uses default priors.

### Prior calibration

```bash
python -m meridian_workflows prior-calibration calibrate
```

Channel0 uses a GeoX result; Channel1 uses generic experimental evidence;
Channel2 combines both; Channel3/4 remain uncalibrated. The notebook's
experiments are **synthetic demonstrations**, not genuine evidence about
your business. This command saves a prior histogram and prior draws without
posterior sampling. Its optional `train` action fits using those priors.

### Full funnel

```bash
python -m meridian_workflows full-funnel prepare
python -m meridian_workflows full-funnel train-stage1 --device cuda --profile smoke
python -m meridian_workflows full-funnel train-stage2 --device cuda --profile smoke
python -m meridian_workflows full-funnel report
# After obtaining adequate fits:
python -m meridian_workflows full-funnel analyze
python -m meridian_workflows full-funnel optimize
```

Stage 1 predicts branded search from awareness advertising. Stage 2 predicts
conversions, with branded search as an unsaturated organic mediator.
Each stage has its own saved model, so a Stage 2 failure does not erase
Stage 1. Use matching chain/draw counts across stages and independent seeds.
`train` can run both stages, but separate stage commands are easier to learn.

The standalone helper retains the notebook's mediator-aware analyzer,
indirect/remaining/total ROI decomposition, and compounded adstock charts.
It depends on the current checkout's internal analysis APIs; update and test
it when upgrading Meridian. The notebook's commented-out, unfinished
full-funnel cloud-upload sketch is not treated as an executable workflow.
The supported Scenario Planner example is the separate entry point below.

### Scenario Planner

```bash
# Compute locally from saved samples. No Google login or upload.
python -m meridian_workflows scenario-planner export \
  --model outputs/getting-started-full/model.binpb

# Optional extra period breakdowns (more expensive):
python -m meridian_workflows scenario-planner export \
  --model outputs/getting-started-full/model.binpb --quarterly
```

This creates `scenario.binpb` and CSVs under `outputs/scenario-planner/tables`.
For the original notebook's model specification, use this demo's own
`prepare`/`train` rather than reusing another model.

Uploading is a separate, explicit action:

```bash
python -m meridian_workflows scenario-planner upload \
  --spreadsheet-name "My Meridian demo"
```

This requires Google Application Default Credentials with both Sheets and
Drive scopes and appropriate API/project permissions. Configure credentials
outside the project; do not commit credentials. The upload creates a Google
spreadsheet and writes its URL and the Scenario Planner link to a text file.
Do not upload private data without authorization. Uploads are mocked in tests.

### MLflow

```bash
python -m meridian_workflows mlflow train --device cuda --profile smoke
python -m meridian_workflows mlflow runs
mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db
```

The demo enables Meridian's autologging, logs the sampling profile, and
stores the model artifact. Tracking is local, using a SQLite database; the
`runs` action does not train. The UI command is optional; stop it with Ctrl+C.

### RF data simulation

```bash
python -m meridian_workflows rf-simulation simulate --seed 1320

# The generated schema can be fed into the RF example.
python -m meridian_workflows reach-frequency prepare \
  --data outputs/rf-simulation/geo_rf_data.csv
```

Outputs: geographic and national CSVs, `ground_truth.npz`,
`raw_parameters.npz`, and time-effect/KPI/spend diagnostic PNGs. This is data
generation, not MCMC.
The simulator preserves the notebook's data-generating equations while using
explicit seeded RNGs for JAX/TF portability; it does not promise the identical
random draws produced by the original TensorFlow seed.

Two deliberate corrections: simulated reach is bounded physically, and
national revenue-per-conversion is conversion-weighted so total revenue is
preserved. National frequency is reach-weighted. Ground truth is stored as
numeric NPZ arrays rather than Python pickle.

## 5. Sampling, performance, and scientific safeguards

The default `smoke` profile uses 2 chains, 200 adaptation steps, 100 burn-in,
200 kept draws, and 200 prior draws. It is only a wiring check.

The `full` profile preserves the notebook-scale total chains/draws. To reduce
peak memory on a 12 GB GPU, chains are grouped in batches of at most two
instead of launching all chains together. This can increase wall time and
may require more adaptation. RF/response/TF/scenario demos use 2,000 adaptation
steps; Getting Started uses 1,000. Full funnel uses 4 chains and 500 kept
draws per stage; MLflow uses 7 chains.

Override parameters explicitly, for example:

```bash
python -m meridian_workflows getting-started train --device cuda \
  --chains 2,2 --adapt 1000 --burnin 500 --keep 1000 --prior 500 \
  --output-dir outputs/diagnostic-candidate
```

`--chains 4` means one batch of four chains; `--chains 2,2` means two batches,
four chains total. No setting guarantees convergence. The profile label
remains `smoke` unless you explicitly choose `--profile full`.

CPU is the default to avoid accidentally starting a large GPU job. CUDA
requests fail explicitly if unavailable; there is no silent CPU fallback.
JAX preallocation is disabled by default, the project's default precision
is preserved, and compilation caching uses a private home-directory path.
Caching reduces repeated compilation, not the amount of MCMC work.

TensorFlow/ArviZ/Altair may still emit upstream warnings. In particular, the
custom ArviZ `trace` warning is not itself a failed fit. Do not suppress
warnings globally or interpret successful file creation as good convergence.

## 6. Tests

```bash
python -m pytest -q
python -m pyink --check meridian_workflows tests
```

Tests force CPU execution. They cover all demo entry points, real example-data
loading, CLI validation, a tiny real posterior-sampling/save/load/report
round trip, response curves, calibrated-prior persistence, full-funnel
algebra, RF simulation invariants, local MLflow wiring, and residual-statistic
values, date ordering, geographic aggregation, and undefined cases. Cloud uploads
are mocked. The tiny sampled model and prior-only full-funnel fixtures are
test data, not scientific validation or production-quality fits.

The examples derive from the repository's Apache-2.0-licensed notebooks;
see the repository [LICENSE](../../LICENSE). Full-funnel helper definitions
retain their original copyright notice.
