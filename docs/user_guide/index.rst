GPClarity User Guide
====================

.. toctree::
   :maxdepth: 2

   getting_started
   gpclarity
   data_influence_map
   hyperparameter_tracker
   examples_gallery

gpclarity.kernel_summary
------------------------

Kernel analysis and interpretation tools.

.. currentmodule:: gpclarity.kernel_summary

.. autofunction:: summarize_kernel

.. autofunction:: interpret_lengthscale

.. autofunction:: interpret_variance

.. autofunction:: format_kernel_tree

.. autofunction:: get_lengthscale

.. autofunction:: get_noise_variance

.. autofunction:: extract_kernel_params_flat

gpclarity.model_complexity
--------------------------

Model complexity and capacity metrics.

.. currentmodule:: gpclarity.model_complexity

.. autofunction:: compute_complexity_score

.. autofunction:: compute_roughness_score

.. autofunction:: compute_noise_ratio

gpclarity.utils
---------------

Model validation and numerical utilities.

.. currentmodule:: gpclarity.utils

.. autofunction:: check_model_health

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

gpclarity.uncertainty_analysis
------------------------------

Uncertainty quantification and calibration.

.. currentmodule:: gpclarity.uncertainty_analysis

.. autoclass:: UncertaintyProfiler
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: quick_uncertainty_check

.. autofunction:: compare_uncertainty_profiles

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

.. autofunction:: plot_uncertainty_profile

.. autofunction:: plot_optimization_trajectory

gpclarity.exceptions
--------------------

Custom exceptions.

.. currentmodule:: gpclarity.exceptions

.. autoexception:: GPClarityError

.. autoexception:: InfluenceError

.. autoexception:: KernelError

.. autoexception:: ComplexityError

.. autoexception:: UncertaintyError

.. autoexception:: TrackingError

.. autoexception:: OptimizationError

.. autoexception:: LinAlgError

.. autoexception:: ValidationError
