# GPClarity: Gaussian Process Interpretability Toolkit

![PyPI](https://img.shields.io/pypi/v/gpclarity)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Build Status](https://github.com/AngadKumar16/gpclarity/workflows/CI/badge.svg)

**GPClarity** is a library that transforms black-box Gaussian Process models into interpretable, debuggable, and trustworthy tools. Built on GPy, it provides human-readable insights into kernel behavior, uncertainty patterns, and model complexity.

---

## Features

- **Kernel Interpretation**: Translate raw kernel math into human-readable summaries with configurable thresholds
- **Uncertainty Profiling**: Visualize, diagnose, and calibrate uncertainty across input space
- **Hyperparameter Tracking**: Monitor optimization trajectories with early stopping and convergence detection
- **Complexity Quantification**: Score model complexity and detect over/underfitting risk
- **Data Influence Analysis**: Identify impactful training points using leverage scores and leave-one-out analysis
- **Model Health Checks**: Validate model state before analysis

---

## Installation

### Stable Release
```bash
pip install gpclarity
```

### Development Version
```bash
git clone https://github.com/AngadKumar16/gpclarity.git
cd gpclarity
pip install -e ".[dev]"
```

### Optional Dependencies

| Package    | Required for                                     |
|------------|--------------------------------------------------|
| `scipy`    | `calibrate_uncertainty()` with scaling method    |
| `sklearn`  | Faster nearest-neighbor search for large datasets |
| `joblib`   | Parallel LOO computation in `DataInfluenceMap`   |
| `tqdm`     | Progress bar in `compute_loo_variance_increase`  |
| `pandas`   | `HyperparameterTracker.to_dataframe()`           |

---

## Quick Start

```python
import gpclarity
import GPy
import numpy as np

# Train a Gaussian Process
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = np.sin(X).flatten() + 0.1 * np.random.randn(50)

kernel = GPy.kern.RBF(1) + GPy.kern.White(1)
model = GPy.models.GPRegression(X, y[:, None], kernel)
model.optimize()

# Check model health
health = gpclarity.check_model_health(model)
print(health["healthy"], health["log_likelihood"])

# Summarize kernel structure
summary = gpclarity.summarize_kernel(model)
# Prints human-readable description of each kernel component

# Profile uncertainty
profiler = gpclarity.UncertaintyProfiler(model, X_train=X)
X_test = np.linspace(-2, 12, 200).reshape(-1, 1)
profiler.plot(X_test, y_train=y)

# Track hyperparameter optimization
tracker = gpclarity.HyperparameterTracker(model)
history = tracker.wrapped_optimize(max_iters=100)
tracker.plot_evolution()

# Compute complexity score
result = gpclarity.compute_complexity_score(model, X)
print(f"Complexity: {result['score']:.2f} — {result['interpretation']}")
```

---

## Module Reference

### `kernel_summary` — Kernel Interpretation

```python
from gpclarity import (
    summarize_kernel,
    format_kernel_tree,
    interpret_lengthscale,
    interpret_variance,
    extract_kernel_params_flat,
    get_lengthscale,
    get_noise_variance,
    count_kernel_components,
)
```

#### `summarize_kernel(model, X=None, verbose=True, config=None)`

Generate a human-readable interpretation of the model's kernel.

```python
summary = gpclarity.summarize_kernel(model)
# Returns dict with kernel_structure, components, overall assessment

# Access component details
for comp in summary["components"]:
    print(comp["type"], comp["params"], comp["interpretation"])
```

Use `config` to customize interpretation thresholds:

```python
from gpclarity.kernel_summary import InterpretationConfig, LengthscaleThresholds, VarianceThresholds

config = InterpretationConfig(
    lengthscale=LengthscaleThresholds(rapid_variation=0.3, smooth_trend=3.0),
    variance=VarianceThresholds(very_low=0.05, high=5.0),
)
summary = gpclarity.summarize_kernel(model, config=config)
```

#### `format_kernel_tree(model)` → `str`

Pretty-print the kernel hierarchy:

```python
print(gpclarity.format_kernel_tree(model))
# Output:
#   └─ rbf
#   └─ white
```

#### `extract_kernel_params_flat(model)` → `dict`

Get all kernel parameters as a flat dict with dotted paths:

```python
params = gpclarity.extract_kernel_params_flat(model)
# {"add.rbf.lengthscale": 1.23, "add.white.variance": 0.01}
```

#### `get_lengthscale(model, as_dict=False)` → `float | dict`

```python
ls = gpclarity.get_lengthscale(model)               # scalar
ls_map = gpclarity.get_lengthscale(model, as_dict=True)  # per-component dict
```

#### `get_noise_variance(model)` → `float`

```python
noise = gpclarity.get_noise_variance(model)
```

---

### `uncertainty_analysis` — Uncertainty Profiling

```python
from gpclarity import UncertaintyProfiler
```

#### `UncertaintyProfiler(model, config=None, X_train=None)`

```python
profiler = UncertaintyProfiler(model, X_train=X)
```

**`predict(X_test, cache_key=None)`** → `PredictionResult`

Safe prediction with variance clipping and caching:

```python
result = profiler.predict(X_test, cache_key="grid")
print(result.mean, result.variance, result.std)

# Get confidence intervals
lower, upper = result.get_interval(level=2.0)  # 2-sigma
```

**`compute_diagnostics(X_test)`** → `dict`

Spatial uncertainty statistics:

```python
diag = profiler.compute_diagnostics(X_test)
# Keys: mean_uncertainty, median_uncertainty, max_uncertainty,
#       min_uncertainty, uncertainty_std, coefficient_of_variation,
#       high_uncertainty_ratio, n_extrapolation_points, ...
```

**`identify_uncertainty_regions(X_test, threshold_percentile=None, return_regions=False)`** → `dict`

```python
regions = profiler.identify_uncertainty_regions(X_test, threshold_percentile=90)
high_pts = regions["high_uncertainty_points"]["points"]
```

**`classify_regions(X_test)`** → `np.ndarray` of `UncertaintyRegion`

Classify each test point as `INTERPOLATION`, `EXTRAPOLATION`, `BOUNDARY`, or `STRUCTURAL`:

```python
from gpclarity.uncertainty_analysis import UncertaintyRegion

labels = profiler.classify_regions(X_test)
n_extrap = np.sum(labels == UncertaintyRegion.EXTRAPOLATION)
```

**`calibrate_uncertainty(X_val, y_val, method="scaling")`** → `dict`

Calibrate uncertainty against validation data:

```python
cal = profiler.calibrate_uncertainty(X_val, y_val)
print(f"Optimal scale: {cal['optimal_scale']:.3f}")
print(f"Original coverage: {cal['original_coverage']:.2%}")
```

**`get_summary(X_test)`** → `dict`

Full summary with diagnostics, region breakdown, and actionable recommendations:

```python
summary = profiler.get_summary(X_test)
for rec in summary["recommendations"]:
    print(rec)
```

**`plot(X_test, X_train=None, y_train=None, confidence_levels=(1.0, 2.0), show_regions=False, ...)`** → `plt.Axes`

```python
ax = profiler.plot(X_test, X_train=X, y_train=y, show_regions=True)
```

**Standalone helpers:**

```python
from gpclarity.uncertainty_analysis import quick_uncertainty_check, compare_uncertainty_profiles

# One-liner
print(quick_uncertainty_check(model, X_test, X_train=X))

# Compare multiple models
results = compare_uncertainty_profiles({"m1": model1, "m2": model2}, X_test)
```

---

### `hyperparam_tracker` — Optimization Tracking

```python
from gpclarity import HyperparameterTracker
```

#### `HyperparameterTracker(model)`

**`wrapped_optimize(max_iters=100, capture_every=1, convergence_tolerance=1e-6, patience=10, callback=None)`** → `dict`

Optimizes with automatic early stopping and full trajectory recording:

```python
tracker = HyperparameterTracker(model)
history = tracker.wrapped_optimize(max_iters=200, patience=15)
# history = {"lengthscale": [1.0, 1.1, ...], "variance": [...]}
```

Use a callback to run custom logic each iteration:

```python
def my_callback(model, iteration, history):
    if iteration % 10 == 0:
        print(f"Iter {iteration}: LL={model.log_likelihood():.3f}")

tracker.wrapped_optimize(max_iters=100, callback=my_callback)
```

**`get_parameter_trajectory(param_name)`** → `(iterations, values)`

```python
iters, vals = tracker.get_parameter_trajectory("lengthscale")
```

**`get_convergence_report(window=10, convergence_threshold=0.01)`** → `dict`

Statistical convergence analysis per parameter:

```python
report = tracker.get_convergence_report()
for param, metrics in report.items():
    print(f"{param}: converged={metrics['converged']}, trend={metrics['trend_direction']}")
```

**`detect_optimization_issues()`** → `dict`

Detect NaN values, parameter oscillation, and decreasing log-likelihood:

```python
issues = tracker.detect_optimization_issues()
for w in issues["warnings"]:
    print(w)
for r in issues["recommendations"]:
    print(r)
```

**`plot_evolution(params=None, show_convergence=True, show_ll=True)`** → `plt.Figure`

```python
fig = tracker.plot_evolution()
```

**`to_dataframe()`** → `pd.DataFrame`

Export full history (requires `pandas`):

```python
df = tracker.to_dataframe()
# Columns: iteration, log_likelihood, gradient_norm, timestamp, <param names>
```

---

### `model_complexity` — Complexity Quantification

```python
from gpclarity import compute_complexity_score, compute_roughness_score, compute_noise_ratio
```

#### `compute_complexity_score(model, X, strategy="default", thresholds=None, return_diagnostics=False)`

Three scoring strategies are available:

| Strategy    | Description                                                |
|-------------|------------------------------------------------------------|
| `"default"` | Composite score from components, roughness, and SNR        |
| `"geometric"`| Effective rank of kernel matrix eigenvalue spectrum       |
| `"bayesian"`| Log marginal likelihood curvature (gradient-based)        |

```python
result = gpclarity.compute_complexity_score(model, X)
print(result["score"])           # raw score
print(result["category"])        # "MODERATE", "COMPLEX", etc.
print(result["interpretation"])  # human-readable description
print(result["risk_level"])      # "LOW", "MEDIUM", "HIGH"
print(result["risk_factors"])    # list of specific risks
print(result["recommendations"]) # actionable suggestions

# Geometric strategy
result = gpclarity.compute_complexity_score(model, X, strategy="geometric")

# Full diagnostics object
from gpclarity.model_complexity import ComplexityMetrics
metrics: ComplexityMetrics = gpclarity.compute_complexity_score(model, X, return_diagnostics=True)
print(metrics.effective_degrees_of_freedom)
print(metrics.capacity_ratio)
print(metrics.is_well_specified)
```

#### `compute_roughness_score(kern)` → `float`

Geometric mean of inverse lengthscales across all kernel components. Higher = more wiggly.

```python
roughness = gpclarity.compute_roughness_score(model.kern)
```

#### `compute_noise_ratio(model)` → `float`

Signal-to-noise ratio (`signal_variance / noise_variance`):

```python
snr = gpclarity.compute_noise_ratio(model)
# >1 means signal dominates, <1 means noise dominates
```

---

### `data_influence` — Data Influence Analysis

```python
from gpclarity import DataInfluenceMap
```

#### `DataInfluenceMap(model)`

**`compute_influence_scores(X_train)`** → `InfluenceResult`

Leverage scores via hat matrix diagonal (O(n³)):

```python
influence = DataInfluenceMap(model)
result = influence.compute_influence_scores(X)

scores = result.scores          # np.ndarray of influence per point
print(result.computation_time)  # seconds
print(result.metadata)          # kernel type, noise variance, etc.

# Find most influential point
top_idx = np.argmax(scores)
print(f"Most influential: point {top_idx} at {X[top_idx]}")
```

**`compute_loo_variance_increase(X_train, y_train, n_jobs=1, verbose=False)`** → `(variance_increase, prediction_errors)`

Exact leave-one-out analysis (slow for large n, supports parallelism via joblib):

```python
var_increase, pred_errors = influence.compute_loo_variance_increase(
    X, y, n_jobs=-1, verbose=True  # use all cores, show tqdm progress
)
# Most impactful point is where var_increase is largest
```

**`get_influence_report(X_train, y_train, compute_loo=True, n_jobs=1)`** → `dict`

Full analysis combining leverage scores and LOO:

```python
report = influence.get_influence_report(X, y)
print(report["most_influential_point"])   # index, location, score
print(report["least_influential_point"])
print(report["diagnostics"]["high_leverage_count"])
print(report["loo_analysis"]["mean_error"])
```

**`plot_influence(X_train, influence_scores, ax=None)`** → `plt.Axes`

```python
result = influence.compute_influence_scores(X)
ax = influence.plot_influence(X, result)
```

---

### `utils` — Model Health

```python
from gpclarity import check_model_health
```

#### `check_model_health(model)` → `dict`

Validate model attributes, parameters, and log-likelihood before analysis:

```python
health = gpclarity.check_model_health(model)
if not health["healthy"]:
    print("Issues:", health["issues"])
print("Warnings:", health["warnings"])
print("Log-likelihood:", health["log_likelihood"])
print("Parameters:", health["n_parameters"])
```

---

## Example Output

### Kernel Summary

```
 KERNEL SUMMARY
==================================================
Structure: ['rbf', 'white']

Thresholds:
  Lengthscale: rapid<0.5, smooth>2.0
  Variance: very_low<0.01, high>10.0

 rbf (parts[0])
  └─ lengthscale: 1.23
    Moderate flexibility (1.23)

 white (parts[1])
  └─ variance: 0.01
    Very low noise (≈0.010)
```

### Complexity Report

```python
{
    "score": 2.34,
    "category": "MODERATE",
    "interpretation": "Well-balanced complexity",
    "risk_level": "LOW",
    "risk_factors": [],
    "recommendations": [],
    "components": {
        "n_parameters": 3,
        "n_kernel_components": 2,
        "roughness_score": 0.81,
        "signal_noise_ratio": 4.5,
        "effective_dof": 12.3,
        "capacity_ratio": 0.25,
    }
}
```

---

## Architecture

```
gpclarity/
├── kernel_summary.py      # Kernel parsing, interpretation, parameter extraction
├── uncertainty_analysis.py # Uncertainty profiling, region classification, calibration
├── hyperparam_tracker.py  # Optimization tracking, convergence detection
├── model_complexity.py    # Complexity scoring (3 strategies), roughness, SNR
├── data_influence.py      # Leverage scores, LOO analysis, influence visualization
├── plotting.py            # Rendering backend for all plot methods
├── utils.py               # Model health checks, numerical utilities
└── exceptions.py          # Custom exception hierarchy
```

---

## Citation

```bibtex
@software{gpclarity2026,
  title={gpclarity: Gaussian Process Interpretability Toolkit},
  author={Angad Kumar},
  year={2026},
  url={https://github.com/AngadKumar16/gpclarity},
  version={0.1.3}
}
```

## License

GPClarity is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome!

- Report bugs or request features via [GitHub Issues](https://github.com/AngadKumar16/gpclarity/issues)
- Submit pull requests for fixes or enhancements
- Run `pip install -e ".[dev]"` and ensure tests pass before submitting

**Author:** Angad Kumar ([GitHub](https://github.com/AngadKumar16), [Email](mailto:angadkumar16ak@gmail.com))

## Roadmap

- Emukit integration: `ClarityBayesianOptimizationLoop` for instrumented Bayesian optimization
- Conda package support
- Automated example notebooks
- Support for GPflow and other GP backends
