Kernel Recommendation
=====================

Picking a kernel is the hardest choice in GP modelling. This module reads the
data first — trend, periodicity, smoothness, noise — and proposes an additive
structure with reasons attached.

Getting a suggestion
--------------------

.. code-block:: python

   import gpclarity

   rec = gpclarity.suggest_kernel(X, y)
   print(rec)

Example output::

   Recommended kernel: Linear + StdPeriodic + White
   Confidence: 78%
   Rationale:
     - Strong linear trend (R²=0.82, slope=+1.94) → add Linear.
     - Dominant cycle detected (strength=0.41, period≈12.1) → add StdPeriodic.
     - Appreciable observation noise (ratio=0.12) → add White.
   Alternatives: Linear + StdPeriodic + Matern52 + White

The recommendation carries the components, a readable ``expression``, a
``confidence`` in ``[0, 1]``, and the :class:`~gpclarity.DataCharacteristics`
the advice was based on. Inspect those characteristics directly when you want to
understand *why*:

.. code-block:: python

   c = gpclarity.analyze_data_characteristics(X, y)
   print(f"trend R²:          {c.trend_strength:.2f}")
   print(f"periodicity:       {c.periodicity_strength:.2f} "
         f"(period ≈ {c.dominant_period})")
   print(f"smoothness:        {c.smoothness:.2f}")
   print(f"noise ratio:       {c.noise_ratio:.2f}")

How the rules work
------------------

The recommender applies four heuristics, in order: a strong linear trend (high
R²) adds a ``Linear`` term; a dominant FFT peak in the detrended residual adds
``StdPeriodic``; the base stationary term is ``RBF`` for smooth data or
``Matern32`` for rougher data; and an appreciable noise ratio adds ``White``.
Thresholds are tunable keyword arguments if your data is unusual.

Building and fitting the kernel
-------------------------------

.. code-block:: python

   import GPy

   kern = gpclarity.build_kernel(rec, input_dim=X.shape[1])
   model = GPy.models.GPRegression(X, y[:, None], kern)
   model.optimize()

   # Close the loop: confirm the suggestion with the interpretability tools
   print(gpclarity.summarize_kernel(model))

Treat the output as a strong starting point, not a final answer — validate the
fitted model with :doc:`predictive_metrics` and :doc:`../api_reference/model_comparison`.
