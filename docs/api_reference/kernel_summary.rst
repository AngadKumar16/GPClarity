Kernel Summary
==============

The ``kernel_summary`` module parses GPy kernel trees into human-readable text.
It translates raw hyperparameter values (lengthscales, variances, noise levels)
into plain-language interpretations and displays composite kernels as an ASCII
hierarchy. Interpretation behaviour is controlled by :class:`InterpretationConfig`,
which lets you adjust the numeric thresholds to match your data's scale.

**When to use:** after training, to understand what structure the kernel has
learned before deploying predictions or debugging unexpected model behaviour.

.. code-block:: python

   import gpclarity

   summary = gpclarity.summarize_kernel(model)
   print(summary["interpretation"])        # plain-language description
   for comp in summary["components"]:
       print(comp["name"], comp["lengthscale_interpretation"])

   gpclarity.format_kernel_tree(model)     # ASCII hierarchy to stdout

.. automodule:: gpclarity.kernel_summary
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Classes
---------------------

.. autosummary::
   :nosignatures:

   LengthscaleThresholds
   VarianceThresholds
   InterpretationConfig

Core Functions
--------------

.. autosummary::
   :nosignatures:

   summarize_kernel
   format_kernel_tree
   interpret_lengthscale
   interpret_variance
   extract_kernel_params
   extract_kernel_params_flat
   get_kernel_structure
   count_kernel_components
   get_lengthscale
   get_noise_variance
