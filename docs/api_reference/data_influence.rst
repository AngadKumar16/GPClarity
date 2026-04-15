Data Influence
==============

The ``data_influence`` module identifies which training points have the greatest
impact on the model's predictions. It computes leverage scores via an O(n³)
Cholesky solve and exact leave-one-out (LOO) variance increases via optional
``joblib`` parallelism. High-leverage points drive the kernel hyperparameters;
high LOO-variance points are informative but hard to interpolate around.

**When to use:** to find outliers that are distorting the fit, remove redundant
training points, or understand which observations are driving predictions in a
given region.

.. code-block:: python

   import gpclarity, numpy as np

   influence = gpclarity.DataInfluenceMap(model)
   result = influence.compute_influence_scores(X_train)
   top_idx = np.argmax(result.scores)
   print(f"Most influential point: index {top_idx}, score {result.scores[top_idx]:.4f}")

   report = influence.get_influence_report(X_train)
   print(report["summary"])

.. automodule:: gpclarity.data_influence
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

.. autosummary::
   :nosignatures:

   DataInfluenceMap
   InfluenceResult
