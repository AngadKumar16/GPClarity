Uncertainty Profiler
====================

Gaussian Processes provide predictive uncertainty as a first-class output —
but raw variance values are hard to interpret. The :class:`~gpclarity.UncertaintyProfiler`
turns them into actionable diagnostics: it classifies regions as interpolation
vs. extrapolation, detects when confidence intervals are too wide or too narrow,
and can recalibrate the uncertainty scale against held-out validation data.

.. contents:: On this page
   :local:
   :depth: 2

Initialization
--------------

.. code-block:: python

   import gpclarity, GPy, numpy as np

   X = np.linspace(0, 10, 50).reshape(-1, 1)
   y = np.sin(X).flatten() + 0.1 * np.random.randn(50)
   model = GPy.models.GPRegression(X, y[:, None], GPy.kern.RBF(1))
   model.optimize()

   profiler = gpclarity.UncertaintyProfiler(model, X_train=X)

Pass ``X_train`` so the profiler can distinguish interpolation from
extrapolation. Providing ``config`` (an :class:`~gpclarity.uncertainty_analysis.UncertaintyConfig`
object) lets you adjust thresholds such as the high-uncertainty percentile cutoff.

Prediction with Intervals
-------------------------

.. code-block:: python

   X_test = np.linspace(-2, 12, 200).reshape(-1, 1)
   result = profiler.predict(X_test)

   print(result.mean.shape)       # (200, 1)
   print(result.variance.shape)   # (200, 1)

   # 2-sigma confidence interval
   lower, upper = result.get_interval(2.0)

``predict()`` returns a :class:`~gpclarity.uncertainty_analysis.PredictionResult`
dataclass. ``get_interval(sigma)`` computes ``mean ± sigma * std``.

Diagnostics
-----------

``compute_diagnostics()`` returns a plain dictionary summarising uncertainty
across the test set:

.. code-block:: python

   diag = profiler.compute_diagnostics(X_test)

   print(diag["mean_uncertainty"])        # average predictive variance
   print(diag["max_uncertainty"])         # peak variance
   print(diag["uncertainty_std"])         # spread across the test set
   print(diag["high_uncertainty_ratio"])  # fraction above the 90th percentile
   print(diag["n_extrapolation_points"])  # points outside the training hull
   print(diag["coefficient_of_variation"])# std / mean — scale-free spread metric

A ``coefficient_of_variation`` between 0.1 and 10.0 is considered well-calibrated.
Values outside that range suggest the uncertainty scale may need adjustment.

Region Classification
---------------------

``classify_regions()`` assigns each test point one of four labels from the
:class:`~gpclarity.uncertainty_analysis.UncertaintyRegion` enum:

.. code-block:: python

   labels = profiler.classify_regions(X_test)
   # Each label is one of:
   # UncertaintyRegion.INTERPOLATION  — inside the training hull, low uncertainty
   # UncertaintyRegion.BOUNDARY       — near the edge of the training data
   # UncertaintyRegion.EXTRAPOLATION  — outside the training hull
   # UncertaintyRegion.STRUCTURAL     — high uncertainty despite dense training data

``identify_uncertainty_regions()`` returns the actual points in each category:

.. code-block:: python

   regions = profiler.identify_uncertainty_regions(X_test, threshold_percentile=90)
   print(regions["high_uncertainty_points"]["points"])
   print(regions["low_uncertainty_points"]["points"])
   print(regions["threshold"])   # variance threshold used

Uncertainty Calibration
-----------------------

If you have held-out validation data with known targets, ``calibrate_uncertainty()``
finds the scalar multiplier that brings the model's uncertainty in line with
observed prediction errors:

.. code-block:: python

   X_val = np.linspace(0, 10, 20).reshape(-1, 1)
   y_val = np.sin(X_val).flatten() + 0.1 * np.random.randn(20)

   cal = profiler.calibrate_uncertainty(X_val, y_val)
   print(cal["optimal_scale"])    # multiply raw std by this factor
   print(cal["coverage_before"])  # empirical 95% coverage before calibration
   print(cal["coverage_after"])   # coverage after applying optimal_scale

A well-calibrated model has ``coverage_after`` close to 0.95 for 2-sigma
intervals. If ``optimal_scale`` >> 1, the model is overconfident; if << 1,
it is underconfident.

Visualization
-------------

.. code-block:: python

   profiler.plot(
       X_test,
       X_train=X, y_train=y,
       confidence_level=2.0,    # number of sigma for shaded band
       show_regions=True,       # colour-code extrapolation regions
   )

The plot shows the posterior mean, confidence band, training data, and
(optionally) a colour overlay for extrapolation regions.

Full Summary
------------

``get_summary()`` combines diagnostics, region classification, and
recommendations into a single dict:

.. code-block:: python

   summary = profiler.get_summary(X_test)

   print(summary["mean_uncertainty"])
   print(summary["n_extrapolation_points"])

   for rec in summary["recommendations"]:
       print("-", rec)

The ``recommendations`` list contains actionable strings such as
*"High extrapolation ratio — restrict predictions to the training domain"*
or *"Model is overconfident — consider calibrating uncertainty scale"*.
