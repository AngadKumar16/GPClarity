# Changelog

All notable changes to GPClarity are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versions follow [Semantic Versioning](https://semver.org/).

---

## [0.1.3] — 2026-04-09

### Added
- `HyperparameterTracker.get_convergence_report()` — per-parameter convergence statistics (CV, trend direction, relative change)
- `HyperparameterTracker.detect_optimization_issues()` — automatic detection of NaN parameters, oscillation, and log-likelihood degradation
- `HyperparameterTracker.to_dataframe()` — export full optimization history to a pandas DataFrame
- `DataInfluenceMap.compute_loo_variance_increase()` — exact leave-one-out variance increase with optional joblib parallelism and tqdm progress
- `DataInfluenceMap.get_influence_report()` — comprehensive influence analysis combining leverage scores and LOO results
- `UncertaintyProfiler.calibrate_uncertainty()` — calibrate uncertainty scale against validation data
- `UncertaintyProfiler.get_summary()` — full uncertainty summary with region breakdown and actionable recommendations
- `compute_complexity_score()` now supports `strategy` parameter: `"default"`, `"geometric"` (eigenvalue effective rank), and `"bayesian"` (gradient-based)
- `compute_complexity_score(return_diagnostics=True)` returns full `ComplexityMetrics` dataclass

### Fixed
- `get_summary()` and `quick_uncertainty_check()` in `uncertainty_analysis.py` now correctly use dict-key access on `compute_diagnostics()` result (previously crashed with `AttributeError`)
- `compute_diagnostics()` return type annotation corrected from `UncertaintyDiagnostics` to `Dict[str, Any]`
- Removed dead duplicate implementations of `count_kernel_components`, `compute_roughness_score`, `compute_noise_ratio`, and `compute_complexity_score` from `utils.py`

### Documentation
- Rewrote `README.md` with full API reference for all public functions and classes
- Fixed RST documentation inaccuracies: wrong dict-vs-attribute access patterns, wrong key names (`suggestions` → `recommendations`, `sigma_scale` → `optimal_scale`)
- Fixed `user_guide/index.rst`: `check_model_health` correctly listed under `gpclarity.utils`
- Improved `gpclarity/__init__.py` module docstring with structured overview of all exported symbols
- Added `CHANGELOG.md` and `CONTRIBUTING.md`

---

## [0.1.2] — 2026

### Added
- `DataInfluenceMap` class with leverage-score influence computation (O(n³) via Cholesky)
- Sphinx documentation structure with Read the Docs theme
- Full `.rst` API reference for all modules
- User guide with getting started, data influence, and hyperparameter tracker pages
- Examples gallery with code snippets for all major features

### Changed
- `HyperparameterTracker.wrapped_optimize()` now supports `capture_every`, `convergence_tolerance`, and `patience` parameters for intelligent early stopping

---

## [0.1.1] — 2026

### Added
- `UncertaintyProfiler.classify_regions()` — classify test points as `INTERPOLATION`, `EXTRAPOLATION`, `BOUNDARY`, or `STRUCTURAL`
- `UncertaintyProfiler.identify_uncertainty_regions()` — identify and characterize high/low uncertainty regions
- `compare_uncertainty_profiles()` — compare uncertainty diagnostics across multiple models
- `quick_uncertainty_check()` — one-line uncertainty assessment string

### Fixed
- Parameter name consistency: `is_healthy` renamed to `healthy` in `check_model_health()` output
- `get_convergence_report()` dict now includes `converged` as alias for `is_converged`

---

## [0.1.0] — 2026

### Added
- Initial release
- `summarize_kernel()` — human-readable kernel interpretation with configurable thresholds
- `format_kernel_tree()` — ASCII kernel hierarchy display
- `interpret_lengthscale()`, `interpret_variance()` — plain-language parameter interpretation
- `extract_kernel_params_flat()`, `get_lengthscale()`, `get_noise_variance()` — parameter extraction utilities
- `UncertaintyProfiler` — uncertainty quantification, prediction with intervals, and visualization
- `HyperparameterTracker` — optimization tracking with trajectory recording and convergence detection
- `compute_complexity_score()` — composite complexity score with interpretation and recommendations
- `compute_roughness_score()`, `compute_noise_ratio()` — component complexity metrics
- `check_model_health()` — model validation before analysis
- `InterpretationConfig`, `LengthscaleThresholds`, `VarianceThresholds` — configurable interpretation thresholds
- Custom exception hierarchy: `GPClarityError`, `InfluenceError`, `KernelError`, `ComplexityError`, `UncertaintyError`, `TrackingError`, `OptimizationError`, `LinAlgError`, `ValidationError`
- Graceful degradation: stub objects with helpful error messages when GPy is not installed
