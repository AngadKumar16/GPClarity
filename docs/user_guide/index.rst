GPClarity User Guide
====================

gpclarity.data_influence
------------------------

Data influence analysis for Gaussian Process models.

.. currentmodule:: gpclarity.data_influence

.. autoclass:: DataInfluenceMap
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: InfluenceResult
   :members:
   :undoc-members:
   :show-inheritance:

gpclarity.kernel_summary
------------------------

Kernel analysis and interpretation tools.

.. currentmodule:: gpclarity.kernel_summary

.. autofunction:: summarize_kernel

.. autofunction:: decompose_kernel_variance

.. autofunction:: check_lengthscale_consistency

.. autofunction:: empirical_kernel_function

gpclarity.model_complexity
--------------------------

Model complexity and capacity metrics.

.. currentmodule:: gpclarity.model_complexity

.. autoclass:: ComplexityAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

gpclarity.uncertainty_analysis
------------------------------

Uncertainty quantification and calibration.

.. currentmodule:: gpclarity.uncertainty_analysis

.. autofunction:: analyze_uncertainty_surface

.. autofunction:: compute_calibration_metrics

gpclarity.hyperparam_tracker
----------------------------

Hyperparameter optimization tracking.

.. currentmodule:: gpclarity.hyperparam_tracker

.. autoclass:: HyperparameterTracker
   :members:
   :undoc-members:
   :show-inheritance:

gpclarity.plotting
------------------

Visualization utilities.

.. currentmodule:: gpclarity.plotting

.. autofunction:: plot_influence_map

.. autofunction:: plot_uncertainty_surface

.. autofunction:: plot_kernel_decomposition

gpclarity.utils
---------------

Utility functions.

.. currentmodule:: gpclarity.utils

.. autofunction:: validate_kernel_matrix

.. autofunction:: cholesky_with_jitter

.. autofunction:: standardize_data

gpclarity.exceptions
--------------------

Custom exceptions.

.. currentmodule:: gpclarity.exceptions

.. autoexception:: InfluenceError

.. autoexception:: KernelError

.. autoexception:: ComplexityError
