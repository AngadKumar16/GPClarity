Utilities
=========

.. automodule:: gpclarity.utils
   :members:
   :undoc-members:
   :show-inheritance:

.. warning::
   These utilities are not part of the stable public API and may change
   without deprecation warnings.

Exceptions
^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ComplexityError
   LinAlgError

Complexity Functions
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   count_kernel_components
   compute_roughness_score
   compute_noise_ratio
   compute_complexity_score
   check_model_health

Validation Functions
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   _validate_array
   _validate_kernel_matrix
   _validate_convergence_window
   _cholesky_with_jitter

Parameter Extraction
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   _extract_param_value
