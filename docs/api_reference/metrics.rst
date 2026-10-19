Predictive Metrics
==================

The ``metrics`` module scores the *quality of predictions* rather than the
model's internal structure. It provides proper scoring rules
(:func:`~gpclarity.nlpd_gaussian`, :func:`~gpclarity.crps_gaussian`,
:func:`~gpclarity.interval_score`) and calibration diagnostics
(:func:`~gpclarity.coverage_probability`, :func:`~gpclarity.pit_values`,
:func:`~gpclarity.calibration_curve`, :func:`~gpclarity.calibration_error`,
:func:`~gpclarity.sharpness`). All scoring functions are pure NumPy and work
without GPy; :func:`~gpclarity.cross_validate` retrains a model per fold and so
needs a working backend at call time.

**When to use:** to decide whether a model's confidence intervals are honest and
to compare candidate models on held-out data.

.. code-block:: python

   import gpclarity

   mean, var = model.predict(X_test)
   scores = gpclarity.compute_all_metrics(y_test, mean, var)
   print(scores["nlpd"], scores["coverage"], scores["calibration_error"])

   # k-fold cross-validation of a model-building function
   def factory(Xtr, ytr):
       import GPy
       m = GPy.models.GPRegression(Xtr, ytr, GPy.kern.RBF(Xtr.shape[1]))
       m.optimize()
       return m

   cv = gpclarity.cross_validate(factory, X, y, n_folds=5)
   print(cv.nlpd, cv.rmse, cv.coverage)

.. automodule:: gpclarity.metrics
   :members:
   :undoc-members:
   :show-inheritance:

Scoring Rules
-------------

.. autosummary::
   :nosignatures:

   nlpd_gaussian
   crps_gaussian
   interval_score

Calibration Diagnostics
-----------------------

.. autosummary::
   :nosignatures:

   coverage_probability
   nominal_coverage
   pit_values
   calibration_curve
   calibration_error
   sharpness

Aggregation & Cross-Validation
------------------------------

.. autosummary::
   :nosignatures:

   compute_all_metrics
   cross_validate
   CVResult
