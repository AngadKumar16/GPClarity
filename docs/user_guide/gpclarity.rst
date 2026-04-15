gpclarity Package
=================

.. currentmodule:: gpclarity

The top-level ``gpclarity`` namespace re-exports all public symbols for convenience.
Import from here rather than from individual submodules.

.. code-block:: python

   import gpclarity

   # Kernel tools
   gpclarity.summarize_kernel(model)
   gpclarity.format_kernel_tree(model)
   gpclarity.interpret_lengthscale(1.5)
   gpclarity.interpret_variance(0.5)
   gpclarity.get_lengthscale(model)
   gpclarity.get_noise_variance(model)
   gpclarity.extract_kernel_params_flat(model)
   gpclarity.count_kernel_components(model.kern)

   # Complexity & health
   gpclarity.compute_complexity_score(model, X)
   gpclarity.compute_roughness_score(model.kern)
   gpclarity.compute_noise_ratio(model)
   gpclarity.check_model_health(model)

   # Classes
   profiler = gpclarity.UncertaintyProfiler(model)
   tracker  = gpclarity.HyperparameterTracker(model)
   influence = gpclarity.DataInfluenceMap(model)

Availability Check
------------------

GPClarity degrades gracefully when GPy is not installed. Stub objects replace all
public symbols and raise an ``ImportError`` with a helpful message on first use.

.. code-block:: python

   import gpclarity
   print(gpclarity.AVAILABLE)
   # {'gpclarity': True, 'full': True}   # when GPy is installed
   # {'gpclarity': True, 'full': False}  # when GPy is missing

Version
-------

.. code-block:: python

   print(gpclarity.__version__)  # e.g. "0.1.3"
