Exceptions
==========

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
