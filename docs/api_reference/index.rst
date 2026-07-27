API Reference
=============

.. module:: gpclarity

Public API
----------

High-level interpretability functions (see the module pages below for full signatures):

.. autosummary::
   :nosignatures:

   summarize_kernel
   interpret_lengthscale
   interpret_variance
   format_kernel_tree
   compute_complexity_score
   compute_roughness_score
   compute_noise_ratio
   count_kernel_components
   check_model_health
   extract_kernel_params_flat
   get_lengthscale
   get_noise_variance

Predictive metrics, calibration, and cross-validation:

.. autosummary::
   :nosignatures:

   nlpd_gaussian
   crps_gaussian
   coverage_probability
   nominal_coverage
   pit_values
   calibration_curve
   calibration_error
   sharpness
   interval_score
   compute_all_metrics
   cross_validate

Kernel recommendation, model comparison, and reporting:

.. autosummary::
   :nosignatures:

   analyze_data_characteristics
   suggest_kernel
   build_kernel
   compare_models
   select_best_model
   score_model
   generate_report

Core Classes
------------

.. autosummary::
   :nosignatures:

   UncertaintyProfiler
   HyperparameterTracker
   DataInfluenceMap
   ModelComparison
   DiagnosticReport
   KernelRecommendation
   CVResult

Module Reference
----------------

.. toctree::
   :maxdepth: 1

   kernel_summary
   uncertainty_analysis
   model_complexity
   hyperparam_tracker
   data_influence
   metrics
   kernel_recommendation
   model_comparison
   reporting
   exceptions
   utils
   plotting
