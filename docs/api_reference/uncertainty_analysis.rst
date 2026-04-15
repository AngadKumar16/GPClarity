Uncertainty Analysis
====================

The ``uncertainty_analysis`` module quantifies, visualises, and calibrates
predictive uncertainty across the input space. :class:`UncertaintyProfiler`
classifies test points as interpolation, extrapolation, or boundary regions,
computes spatial uncertainty statistics, and can scale uncertainty estimates
to match observed coverage on validation data.

**When to use:** before deploying predictions, to identify where the model is
unreliable and whether its confidence intervals are well-calibrated.

.. code-block:: python

   import gpclarity, numpy as np

   profiler = gpclarity.UncertaintyProfiler(model, X_train=X_train)
   diag = profiler.compute_diagnostics(X_test)
   print(f"Mean uncertainty: {diag['mean_uncertainty']:.4f}")
   print(f"Extrapolation points: {diag['n_extrapolation_points']}")

   summary = profiler.get_summary(X_test)
   for rec in summary["recommendations"]:
       print("-", rec)

.. automodule:: gpclarity.uncertainty_analysis
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

.. autosummary::
   :nosignatures:

   UncertaintyProfiler
   UncertaintyConfig
   UncertaintyDiagnostics
   PredictionResult
   UncertaintyRegion

Functions
---------

.. autosummary::
   :nosignatures:

   quick_uncertainty_check
   compare_uncertainty_profiles