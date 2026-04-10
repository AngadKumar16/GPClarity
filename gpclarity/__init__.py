"""
gpclarity: Interpretability Toolkit for Gaussian Processes
===========================================================

GPClarity makes Gaussian Process models transparent and debuggable by
providing human-readable diagnostics across five areas:

Kernel Interpretation
---------------------
- ``summarize_kernel(model)``           — structured human-readable kernel summary
- ``format_kernel_tree(model)``         — ASCII tree of kernel hierarchy
- ``interpret_lengthscale(ls)``         — plain-language lengthscale interpretation
- ``interpret_variance(var)``           — plain-language variance interpretation
- ``extract_kernel_params_flat(model)`` — all hyperparameters as a flat dict
- ``get_lengthscale(model)``            — scalar or per-component lengthscale
- ``get_noise_variance(model)``         — noise variance scalar
- ``count_kernel_components(kern)``     — count of leaf kernel components

Uncertainty Analysis
--------------------
- ``UncertaintyProfiler(model)``        — profile, calibrate, and visualize uncertainty
  - ``.predict(X_test)``               → PredictionResult with mean, variance, intervals
  - ``.compute_diagnostics(X_test)``   → dict of spatial uncertainty statistics
  - ``.identify_uncertainty_regions()``→ high/low uncertainty point sets
  - ``.classify_regions(X_test)``      → per-point INTERPOLATION/EXTRAPOLATION labels
  - ``.calibrate_uncertainty()``       → scale uncertainty to match validation coverage
  - ``.get_summary(X_test)``           → full summary with recommendations
  - ``.plot(X_test)``                  → confidence band plot

Hyperparameter Tracking
-----------------------
- ``HyperparameterTracker(model)``      — track optimization trajectories
  - ``.wrapped_optimize()``            → optimize with early stopping + history
  - ``.get_convergence_report()``      → per-parameter convergence statistics
  - ``.detect_optimization_issues()``  → NaN, oscillation, and LL degradation checks
  - ``.plot_evolution()``              → parameter trajectory plots
  - ``.to_dataframe()``               → export history to pandas DataFrame

Model Complexity
----------------
- ``compute_complexity_score(model, X)``— composite score with category and recommendations
- ``compute_roughness_score(kern)``     — inverse-lengthscale roughness metric
- ``compute_noise_ratio(model)``        — signal-to-noise ratio

Data Influence
--------------
- ``DataInfluenceMap(model)``           — identify high-leverage training points
  - ``.compute_influence_scores()``    → leverage-score influence per point
  - ``.compute_loo_variance_increase()``→ exact leave-one-out variance impact
  - ``.get_influence_report()``        → comprehensive influence analysis report
  - ``.plot_influence()``              → scatter visualization of influence scores

Utilities
---------
- ``check_model_health(model)``         — validate model before analysis

Quick start::

    import gpclarity, GPy, numpy as np

    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = np.sin(X).flatten() + 0.1 * np.random.randn(50)
    model = GPy.models.GPRegression(X, y[:, None], GPy.kern.RBF(1))
    model.optimize()

    gpclarity.summarize_kernel(model)
    profiler = gpclarity.UncertaintyProfiler(model, X_train=X)
    profiler.plot(np.linspace(-2, 12, 200).reshape(-1, 1), X_train=X, y_train=y)
"""

import warnings
from importlib.util import find_spec

from ._version import __version__

_GPY_AVAILABLE = find_spec("GPy") is not None

AVAILABLE = {
    "gpy": _GPY_AVAILABLE,
    "full": _GPY_AVAILABLE,
}

__all__ = [
    "__version__",
    "AVAILABLE",
    "summarize_kernel",
    "format_kernel_tree",
    "interpret_lengthscale",
    "interpret_variance",
    "UncertaintyProfiler",
    "HyperparameterTracker",
    "compute_complexity_score",
    "count_kernel_components",
    "compute_roughness_score",
    "compute_noise_ratio",
    "DataInfluenceMap",
    "check_model_health",
    "extract_kernel_params_flat",
    "get_lengthscale",
    "get_noise_variance",
]

if _GPY_AVAILABLE:
    from .kernel_summary import (
        summarize_kernel,
        format_kernel_tree,
        interpret_lengthscale,
        extract_kernel_params_flat,
        interpret_variance,
        get_lengthscale,
        get_noise_variance,
        count_kernel_components,
    )
    from .uncertainty_analysis import UncertaintyProfiler
    from .hyperparam_tracker import HyperparameterTracker
    from .model_complexity import (
        compute_complexity_score,
        compute_roughness_score,
        compute_noise_ratio,
    )
    from .data_influence import DataInfluenceMap
    from .utils import (
        check_model_health,
    )
else:
    # Define stubs
    class _Stub:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args, **kwargs):
            raise ImportError(
                f"{self.name} requires GPy. Install: pip install gpclarity[full]"
            )

    summarize_kernel = _Stub("summarize_kernel")
    format_kernel_tree = _Stub("format_kernel_tree")
    interpret_lengthscale = _Stub("interpret_lengthscale")
    interpret_variance = _Stub("interpret_variance")
    UncertaintyProfiler = _Stub("UncertaintyProfiler")
    HyperparameterTracker = _Stub("HyperparameterTracker")
    compute_complexity_score = _Stub("compute_complexity_score")
    count_kernel_components = _Stub("count_kernel_components")
    compute_roughness_score = _Stub("compute_roughness_score")
    compute_noise_ratio = _Stub("compute_noise_ratio")
    DataInfluenceMap = _Stub("DataInfluenceMap")
    check_model_health = _Stub("check_model_health")
    extract_kernel_params_flat = _Stub("extract_kernel_params_flat")
    get_lengthscale = _Stub("get_lengthscale")
    get_noise_variance = _Stub("get_noise_variance")

    warnings.warn(
        "gpclarity running in limited mode. Install with 'pip install gpclarity[full]'",
        ImportWarning,
    )

__author__ = "Angad Kumar"
__email__ = "angadkumar16ak@gmail.com"
