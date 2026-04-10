Getting Started
===============

This guide walks you through the core GPClarity workflow using a simple example.

Setting Up a GP Model
---------------------

GPClarity works with any GPy ``GPRegression`` model. Start by training a model:

.. code-block:: python

   import numpy as np
   import GPy
   import gpclarity

   np.random.seed(42)
   X = np.linspace(0, 10, 50).reshape(-1, 1)
   y = np.sin(X).flatten() + 0.1 * np.random.randn(50)

   kernel = GPy.kern.RBF(1) + GPy.kern.White(1)
   model = GPy.models.GPRegression(X, y[:, None], kernel)
   model.optimize()

Checking Model Health
---------------------

Before analysis, verify the model is well-configured:

.. code-block:: python

   health = gpclarity.check_model_health(model)
   print(f"Healthy: {health['healthy']}")
   for issue in health['issues']:
       print(f"  Issue: {issue}")
   for warning in health['warnings']:
       print(f"  Warning: {warning}")

Interpreting the Kernel
-----------------------

Translate the kernel hyperparameters into plain language:

.. code-block:: python

   summary = gpclarity.summarize_kernel(model, verbose=True)
   # Prints a structured report of each kernel component

The returned dictionary contains:

- ``kernel_structure``: Nested list or string describing kernel composition
- ``components``: List of component dicts with ``params`` and ``interpretation``
- ``composite``: Boolean flag for composite kernels
- ``overall``: High-level assessment string

Analyzing Uncertainty
---------------------

Create an ``UncertaintyProfiler`` to examine where the model is confident:

.. code-block:: python

   profiler = gpclarity.UncertaintyProfiler(model, X_train=X)

   X_test = np.linspace(-2, 12, 200).reshape(-1, 1)

   # Summarize uncertainty behavior
   diagnostics = profiler.compute_diagnostics(X_test)
   print(f"Mean uncertainty: {diagnostics['mean_uncertainty']:.4f}")
   print(f"Extrapolation points: {diagnostics['n_extrapolation_points']}")
   cv = diagnostics['coefficient_of_variation']
   print(f"Well calibrated: {0.1 < cv < 10.0}")

   # Plot with confidence intervals
   profiler.plot(X_test, X_train=X, y_train=y)

Tracking Optimization
---------------------

Monitor hyperparameter convergence during training:

.. code-block:: python

   tracker = gpclarity.HyperparameterTracker(model)

   # Re-optimize with tracking
   history = tracker.wrapped_optimize(max_iters=100, patience=15)

   # Check convergence
   report = tracker.get_convergence_report(window=10)
   for param, metrics in report.items():
       status = "converged" if metrics['is_converged'] else metrics['trend_direction']
       print(f"  {param}: {status}")

   # Visualize trajectories
   fig = tracker.plot_evolution()

Measuring Complexity
--------------------

Assess whether your model is appropriately complex:

.. code-block:: python

   report = gpclarity.compute_complexity_score(model, X)
   print(f"Score: {report['score']:.2f}  ({report['interpretation']})")
   print(f"Components: {report['components']}")

   for rec in report['recommendations']:
       print(f"  Recommendation: {rec}")

Analyzing Data Influence
------------------------

Find which training points matter most to predictions:

.. code-block:: python

   influence = gpclarity.DataInfluenceMap(model)

   # Fast leverage scores
   result = influence.compute_influence_scores(X)
   top_idx = np.argmax(result.scores)
   print(f"Most influential point: index {top_idx} at X={X[top_idx, 0]:.2f}")

   # Full report with leave-one-out analysis
   report = influence.get_influence_report(X, y)
   print(f"High-leverage points: {report['diagnostics']['high_leverage_count']}")

   # Visualize
   influence.plot_influence(X, result)

Next Steps
----------

- See :doc:`data_influence_map` for a detailed guide on influence analysis.
- See :doc:`hyperparameter_tracker` for advanced tracking options.
- See :doc:`../api_reference/index` for full API documentation.
