Kernel Recommendation
=====================

The ``kernel_recommendation`` module inspects raw data — its trend, periodicity,
smoothness, and noise — and suggests an additive kernel structure with a
plain-language rationale, so you start modelling from an informed guess instead
of a bare ``RBF``. :func:`~gpclarity.analyze_data_characteristics` and
:func:`~gpclarity.suggest_kernel` are pure NumPy;
:func:`~gpclarity.build_kernel` turns a suggestion into a concrete ``GPy.kern``
object and needs GPy.

**When to use:** at the very start of a modelling task, before you have committed
to a kernel, or when an ``RBF`` fit looks wrong and you want a data-driven
second opinion.

.. code-block:: python

   import gpclarity

   rec = gpclarity.suggest_kernel(X, y)
   print(rec)                     # human-readable structure + rationale
   print(rec.expression)          # e.g. "Linear + StdPeriodic + White"

   # Build the recommended kernel and fit it
   kern = gpclarity.build_kernel(rec, input_dim=X.shape[1])
   import GPy
   model = GPy.models.GPRegression(X, y[:, None], kern)
   model.optimize()

.. automodule:: gpclarity.kernel_recommendation
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autosummary::
   :nosignatures:

   analyze_data_characteristics
   suggest_kernel
   build_kernel

Data Classes
------------

.. autosummary::
   :nosignatures:

   DataCharacteristics
   KernelRecommendation
