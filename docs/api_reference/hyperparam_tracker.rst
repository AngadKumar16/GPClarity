Hyperparameter Tracker
======================

The ``hyperparam_tracker`` module wraps GPy's optimizer to record a snapshot of
all hyperparameters and the log-likelihood at each iteration. It detects
convergence issues (oscillation, NaN parameters, likelihood degradation) and
exports the full trajectory to a pandas DataFrame for offline analysis.

**When to use:** when you want to understand optimization dynamics — not just the
final parameter values — or diagnose a model that isn't converging cleanly.

.. code-block:: python

   import gpclarity

   tracker = gpclarity.HyperparameterTracker(model)
   tracker.wrapped_optimize(max_iters=200, capture_every=5)

   report = tracker.get_convergence_report()
   print(report["converged"], report["n_iterations"])

   issues = tracker.detect_optimization_issues()
   if issues["has_issues"]:
       print(issues["summary"])

   df = tracker.to_dataframe()   # full history as a pandas DataFrame

.. automodule:: gpclarity.hyperparam_tracker
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

.. autosummary::
   :nosignatures:

   HyperparameterTracker
   OptimizationState
   ConvergenceMetrics
