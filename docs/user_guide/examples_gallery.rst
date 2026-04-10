Examples Gallery
================

The examples below build on the quick-start workflow. Each snippet assumes
the model and data from :doc:`getting_started` are already in scope.

Composite Kernel Inspection
----------------------------

Decompose a sum kernel and inspect each component individually:

.. code-block:: python

   import GPy
   import numpy as np
   import gpclarity

   kernel = GPy.kern.RBF(1) + GPy.kern.Periodic(1) + GPy.kern.White(1)
   model = GPy.models.GPRegression(X, y[:, None], kernel)
   model.optimize()

   # Tree structure
   print(gpclarity.format_kernel_tree(model))

   # Full flat parameter dict
   params = gpclarity.extract_kernel_params_flat(model)
   for name, value in params.items():
       print(f"  {name}: {value:.4f}")

   # Summary with custom thresholds
   from gpclarity.kernel_summary import InterpretationConfig, LengthscaleThresholds
   cfg = InterpretationConfig(
       lengthscale=LengthscaleThresholds(rapid_variation=0.3, smooth_trend=3.0)
   )
   gpclarity.summarize_kernel(model, config=cfg)

Uncertainty Region Classification
-----------------------------------

Identify extrapolation and high-uncertainty regions:

.. code-block:: python

   profiler = gpclarity.UncertaintyProfiler(model, X_train=X)
   X_test = np.linspace(-5, 15, 300).reshape(-1, 1)

   regions = profiler.classify_regions(X_test, X_train=X)

   from gpclarity.uncertainty_analysis import UncertaintyRegion
   extrap_mask = regions == UncertaintyRegion.EXTRAPOLATION
   print(f"Extrapolation: {extrap_mask.sum()} / {len(X_test)} points")

Uncertainty Calibration
-----------------------

Calibrate model uncertainty using a held-out validation set:

.. code-block:: python

   X_val = np.linspace(0, 10, 20).reshape(-1, 1)
   y_val = np.sin(X_val).flatten() + 0.1 * np.random.randn(20)

   cal = profiler.calibrate_uncertainty(X_val, y_val, method="scaling")
   print(f"Optimal sigma scale: {cal.get('optimal_scale', 1.0):.3f}")
   print(f"Miscalibration: {cal.get('miscalibration', 0.0):.4f}")

Comparing Multiple Models
--------------------------

Rank models by uncertainty quality side-by-side:

.. code-block:: python

   from gpclarity.uncertainty_analysis import compare_uncertainty_profiles

   models = {
       "RBF": model_rbf,
       "Matern32": model_mat32,
       "RBF+Periodic": model_composite,
   }

   profiles = compare_uncertainty_profiles(models, X_test, X_train=X)
   for name, diag in profiles.items():
       print(f"{name}: mean_var={diag['mean_uncertainty']:.4f}, CV={diag['coefficient_of_variation']:.2f}")

Optimization Issue Detection
-----------------------------

Automatically detect and diagnose convergence problems:

.. code-block:: python

   tracker = gpclarity.HyperparameterTracker(model)
   tracker.wrapped_optimize(max_iters=200)

   issues = tracker.detect_optimization_issues()
   for w in issues['warnings']:
       print(f"Warning: {w}")
   for r in issues['recommendations']:
       print(f"Tip: {r}")

Exporting Tracker History
--------------------------

Export the optimization trajectory to a CSV file for offline analysis:

.. code-block:: python

   df = tracker.to_dataframe()
   print(df.head())
   df.to_csv("optimization_history.csv", index=False)

Parallel Leave-One-Out Analysis
--------------------------------

Speed up LOO variance computation on large datasets using all CPU cores:

.. code-block:: python

   influence = gpclarity.DataInfluenceMap(model)
   var_increase, pred_errors = influence.compute_loo_variance_increase(
       X, y, n_jobs=-1, verbose=True
   )

   # Points with both high variance increase AND high prediction error are outliers
   threshold = np.percentile(var_increase, 90)
   outlier_mask = var_increase > threshold
   print(f"Potential outliers: {outlier_mask.sum()} points")
