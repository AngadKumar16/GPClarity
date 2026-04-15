Utilities
=========

The ``utils`` module provides shared numerical helpers used internally by all
GPClarity modules, plus :func:`check_model_health` — the only public, stable
export from this module. Call ``check_model_health`` before running any analysis
to confirm that the model has finite parameters and a computable log-likelihood.

Private helpers (``_cholesky_with_jitter``, ``_validate_array``, etc.) are
implementation details and may change without a deprecation notice.

.. code-block:: python

   import gpclarity

   health = gpclarity.check_model_health(model)
   if not health["healthy"]:
       for issue in health["issues"]:
           print("ERROR:", issue)
   for w in health["warnings"]:
       print("WARN:", w)

.. automodule:: gpclarity.utils
   :members:
   :undoc-members:
   :show-inheritance:

Public API
^^^^^^^^^^

.. autosummary::
   :nosignatures:

   check_model_health

Internal Helpers
^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   _validate_array
   _validate_kernel_matrix
   _validate_convergence_window
   _cholesky_with_jitter
   _extract_param_value
