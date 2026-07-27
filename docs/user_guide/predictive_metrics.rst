Predictive Metrics & Calibration
================================

A GP gives you a mean *and* a variance at every point. These tools tell you
whether that variance is trustworthy and let you compare models on held-out
data. Everything except :func:`~gpclarity.cross_validate` is pure NumPy and works
on any ``(y_true, mean, variance)`` triple.

Scoring a single prediction set
--------------------------------

.. code-block:: python

   import gpclarity

   mean, var = model.predict(X_test)
   scores = gpclarity.compute_all_metrics(y_test, mean, var)

   print(f"NLPD:              {scores['nlpd']:.3f}")   # lower is better
   print(f"CRPS:              {scores['crps']:.3f}")   # lower is better
   print(f"RMSE:              {scores['rmse']:.3f}")
   print(f"Coverage (95%):    {scores['coverage']:.2%} "
         f"(target {scores['nominal_coverage']:.2%})")
   print(f"Calibration error: {scores['calibration_error']:.3f}")
   print(f"Sharpness:         {scores['sharpness']:.3f}")

Two ideas drive all of this. **Calibration** asks whether stated intervals are
honest — a 95% interval should contain the truth about 95% of the time.
**Sharpness** asks how tight the intervals are. The goal is the sharpest model
that is still calibrated, so always read a sharpness number next to a
calibration number.

Reading a calibration curve
---------------------------

:func:`~gpclarity.calibration_curve` sweeps nominal confidence levels and reports
the observed coverage at each. A perfectly calibrated model lies on the diagonal;
points below it mean the model is over-confident.

.. code-block:: python

   curve = gpclarity.calibration_curve(y_test, mean, var, n_bins=10)

   import matplotlib.pyplot as plt
   plt.plot([0, 1], [0, 1], "k--", label="perfect")
   plt.plot(curve["expected"], curve["observed"], "o-", label="model")
   plt.xlabel("expected coverage"); plt.ylabel("observed coverage")
   plt.legend()

The Probability Integral Transform gives a complementary view: for a calibrated
model :func:`~gpclarity.pit_values` are uniform on ``[0, 1]``. A PIT histogram
that bulges in the middle means under-confident predictions; a U-shape means
over-confident.

Cross-validation
----------------

To estimate out-of-sample quality without a dedicated test set, pass a factory
that *builds and trains* a fresh model to :func:`~gpclarity.cross_validate`.

.. code-block:: python

   import GPy

   def factory(X_train, y_train):
       m = GPy.models.GPRegression(X_train, y_train, GPy.kern.RBF(X_train.shape[1]))
       m.optimize()
       return m

   cv = gpclarity.cross_validate(factory, X, y, n_folds=5, random_state=0)
   print(cv.nlpd, cv.crps, cv.rmse, cv.coverage)
   for fold in cv.per_fold:
       print(fold)
