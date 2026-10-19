Exceptions
==========

All GPClarity exceptions inherit from :class:`~gpclarity.exceptions.GPClarityError`. Catch it to
handle any library error in a single ``except`` block, or catch a more specific
subclass when you want targeted recovery.

.. code-block:: python

   from gpclarity.exceptions import GPClarityError, InfluenceError

   try:
       result = influence.compute_loo_variance_increase(X_train)
   except InfluenceError as e:
       print(f"LOO computation failed: {e}")
   except GPClarityError as e:
       print(f"Unexpected GPClarity error: {e}")

.. automodule:: gpclarity.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Exception Hierarchy
-------------------

.. code-block:: text

   GPClarityError
   ├── KernelError
   ├── ModelError
   ├── ComplexityError
   ├── UncertaintyError
   ├── TrackingError
   ├── OptimizationError
   ├── InfluenceError
   ├── LinAlgError
   └── ValidationError

All exceptions inherit from ``GPClarityError``, which itself inherits from Python's built-in
``Exception``. Catch ``GPClarityError`` to handle any library error in a single except block.

.. autosummary::
   :nosignatures:

   GPClarityError
   KernelError
   ModelError
   ComplexityError
   UncertaintyError
   TrackingError
   OptimizationError
   InfluenceError
   LinAlgError
   ValidationError
