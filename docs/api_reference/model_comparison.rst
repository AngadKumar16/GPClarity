Model Comparison
================

The ``model_comparison`` module ranks several trained GP models of the same data
on a common footing: marginal likelihood, information criteria
(AIC/BIC), and — when held-out data is supplied — predictive metrics such as
NLPD, CRPS, RMSE, and calibration error. It consumes models by duck typing
(``log_likelihood()`` and ``predict()``), so it imports without GPy.

**When to use:** when you have two or more candidate kernels/models and need a
defensible, quantitative reason to pick one.

.. code-block:: python

   import gpclarity

   models = {"rbf": m_rbf, "matern32": m_mat32, "rbf+periodic": m_comp}

   # Rank by BIC (penalises extra parameters)
   cmp = gpclarity.compare_models(models, criterion="bic")
   print(cmp.best)
   print(cmp.to_table())

   # Or rank by held-out predictive quality
   cmp = gpclarity.compare_models(
       models, X_test=X_test, y_test=y_test, criterion="nlpd"
   )
   name, best_model = gpclarity.select_best_model(models, criterion="bic")

.. automodule:: gpclarity.model_comparison
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autosummary::
   :nosignatures:

   compare_models
   select_best_model
   score_model

Data Classes
------------

.. autosummary::
   :nosignatures:

   ModelComparison
   ModelScore
