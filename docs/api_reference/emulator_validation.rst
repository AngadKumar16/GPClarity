Emulator Validation
===================

The ``emulator_validation`` module implements the GP emulator diagnostics of
Bastos & O'Hagan (2009), *Diagnostics for Gaussian Process Emulators*,
Technometrics 51(4), 425-438. It compares the model's **joint** predictive
distribution against held-out validation data, so it accounts for correlation
between predictions that per-point calibration metrics ignore.

* **Standardized errors**: each residual divided by its predictive standard
  deviation. Roughly standard normal for a valid emulator.
* **Mahalanobis distance**: one summary of all residuals, compared with a
  chi-squared reference (one degree of freedom per validation point). Too large
  means the model is overconfident; too small means it is underconfident.
* **Pivoted Cholesky errors**: residuals decorrelated in order of decreasing
  predictive uncertainty. Large errors late in the order may indicate a
  misspecified correlation structure, such as lengthscales that are too long.

**When to use:** after fitting, on validation points that were not used for
training, to check whether the GP's error bars can be trusted.

.. code-block:: python

   import gpclarity

   result = gpclarity.validate_emulator(model, X_val, y_val)
   print(result.verdict)               # consistent / overconfident / underconfident
   print(result.mahalanobis, result.mahalanobis_expected, result.mahalanobis_pvalue)
   for note in result.interpretation:
       print("-", note)

   # Any joint prediction works, not only GPy models
   mean, cov = other_model.predict(X_val, full_cov=True)
   result = gpclarity.emulator_diagnostics(y_val, mean, cov)

.. note::

   The chi-squared reference treats hyperparameters as known. With estimated
   hyperparameters the exact reference is an F distribution; treat borderline
   p-values with care.

.. automodule:: gpclarity.emulator_validation
   :members:
   :undoc-members:
   :show-inheritance:
