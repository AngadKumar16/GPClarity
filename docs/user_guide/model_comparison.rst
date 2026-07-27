Model Comparison
================

When you have several candidate models, this module ranks them on a common
footing and picks a winner, so the choice is quantitative rather than a guess.

Ranking by information criterion
--------------------------------

.. code-block:: python

   import gpclarity

   models = {
       "rbf": model_rbf,
       "matern32": model_matern,
       "rbf+periodic": model_composite,
   }

   cmp = gpclarity.compare_models(models, criterion="bic")
   print("Best:", cmp.best)
   print(cmp.to_table())

BIC and AIC both reward fit (high marginal likelihood) while penalising extra
parameters; BIC penalises complexity more heavily as the dataset grows. Use them
when you have no separate test set. ``criterion="log_likelihood"`` ignores the
complexity penalty and simply prefers the best in-sample fit.

Ranking on held-out data
-------------------------

With a test split you can rank by predictive quality instead, reusing the metrics
from :doc:`predictive_metrics`.

.. code-block:: python

   cmp = gpclarity.compare_models(
       models, X_test=X_test, y_test=y_test, criterion="nlpd"
   )
   for score in cmp.scores:
       print(score.name, score.test_metrics)

Valid test-set criteria are ``"nlpd"``, ``"crps"``, ``"rmse"``, and
``"calibration_error"`` (all lower-is-better).

Picking the winner directly
----------------------------

.. code-block:: python

   name, best_model = gpclarity.select_best_model(models, criterion="bic")
   print(f"Selected {name}")

Exporting the comparison
------------------------

.. code-block:: python

   df = cmp.to_dataframe()     # pandas, indexed by rank
   payload = cmp.to_dict()     # JSON-ready
