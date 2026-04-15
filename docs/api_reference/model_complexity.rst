Model Complexity
================

The ``model_complexity`` module scores a GP model's structural complexity using
three complementary strategies: ``"default"`` (composite inverse-lengthscale and
noise-ratio heuristic), ``"geometric"`` (effective rank via kernel eigenvalues),
and ``"bayesian"`` (gradient-based sensitivity). A high score signals an
over-parameterised kernel; a low score may indicate under-fitting.

**When to use:** after training, to audit kernel complexity before deployment or
to compare model variants (simpler vs. composite kernels).

.. code-block:: python

   import gpclarity

   result = gpclarity.compute_complexity_score(model, X, strategy="geometric")
   print(result["score"], result["category"], result["recommendations"])

   # Full diagnostics dataclass
   metrics = gpclarity.compute_complexity_score(
       model, X, return_diagnostics=True
   )
   print(metrics.roughness_score, metrics.noise_ratio)

.. automodule:: gpclarity.model_complexity
   :members:
   :undoc-members:
   :show-inheritance:

Enums & Configuration
---------------------

.. autosummary::
   :nosignatures:

   ComplexityCategory
   ComplexityThresholds

Data Classes
------------

.. autosummary::
   :nosignatures:

   ComplexityMetrics

Classes
-------

.. autosummary::
   :nosignatures:

   ComplexityAnalyzer

Functions
---------

.. autosummary::
   :nosignatures:

   compute_complexity_score
   compute_roughness_score
   compute_noise_ratio
   quick_complexity_check
