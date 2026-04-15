Kernel Interpretation
=====================

A GP's kernel encodes prior beliefs about the smoothness, periodicity, and
scale of the function being modelled. GPy stores these beliefs as hyperparameter
objects on a ``model.kern`` tree — but a raw lengthscale of ``2.7`` or a variance
of ``0.04`` is not immediately meaningful. The ``kernel_summary`` module
translates these values into plain-language descriptions and displays composite
kernels as an ASCII hierarchy.

.. contents:: On this page
   :local:
   :depth: 2

Basic Summary
-------------

.. code-block:: python

   import gpclarity

   summary = gpclarity.summarize_kernel(model)

   print(summary["interpretation"])   # e.g. "Smooth, long-range dependencies"
   print(summary["n_components"])     # number of leaf kernels

   for comp in summary["components"]:
       print(comp["name"])
       print(comp["lengthscale_interpretation"])
       print(comp["variance_interpretation"])

The returned dictionary always contains ``"interpretation"``, ``"components"``,
``"n_components"``, and ``"kernel_type"``. Each component dict includes the raw
hyperparameter values alongside their plain-language counterparts.

Kernel Tree
-----------

For composite kernels (sums, products), ``format_kernel_tree()`` prints an ASCII
hierarchy showing how the components are combined:

.. code-block:: python

   gpclarity.format_kernel_tree(model)

Example output for an ``RBF + Matern52`` kernel::

   RBF
   Matern52

For deeply nested products and sums::

   Add
   ├── RBF
   └── Mul
       ├── Matern52
       └── Periodic

Component Interpretation
------------------------

The two interpretation helpers work on scalar values directly, which is useful
when scanning a parameter sweep or comparing models:

.. code-block:: python

   ls_text = gpclarity.interpret_lengthscale(2.7)
   # e.g. "Moderate lengthscale — captures medium-range structure"

   var_text = gpclarity.interpret_variance(0.04)
   # e.g. "Low variance — model explains relatively little signal"

Default thresholds classify lengthscales as "very short" (< 0.1), "short"
(0.1–0.5), "moderate" (0.5–2.0), "long" (2.0–10.0), or "very long" (> 10.0).
Variance uses analogous bands. If your data is on a different scale — say,
time in years rather than seconds — you should adjust these thresholds (see
below).

Parameter Extraction
--------------------

``extract_kernel_params_flat()`` returns all hyperparameters as a flat
dictionary keyed by ``"<kernel_name>.<param_name>"``:

.. code-block:: python

   params = gpclarity.extract_kernel_params_flat(model)
   # e.g. {"rbf.variance": 1.0, "rbf.lengthscale": 2.7, "Gaussian_noise.variance": 0.01}

   for name, value in params.items():
       print(f"{name}: {value:.4f}")

For quick access to the most common values:

.. code-block:: python

   ls    = gpclarity.get_lengthscale(model)     # scalar or array for ARD kernels
   noise = gpclarity.get_noise_variance(model)  # Gaussian noise variance

   n = gpclarity.count_kernel_components(model.kern)
   print(f"Kernel has {n} leaf component(s)")

Custom Thresholds
-----------------

If your input domain spans a very different range than the defaults assume
(e.g. inputs in the range 0–1000 rather than 0–10), the default threshold
bands will label nearly every lengthscale as "very short". Override them with
:class:`~gpclarity.kernel_summary.InterpretationConfig`:

.. code-block:: python

   from gpclarity.kernel_summary import (
       InterpretationConfig, LengthscaleThresholds, VarianceThresholds
   )

   config = InterpretationConfig(
       lengthscale=LengthscaleThresholds(
           very_short=10.0,
           short=50.0,
           moderate=200.0,
           long=1000.0,
       ),
       variance=VarianceThresholds(
           very_low=0.01,
           low=0.1,
           moderate=1.0,
           high=10.0,
       ),
   )

   summary = gpclarity.summarize_kernel(model, config=config)

Pass the same ``config`` to :func:`~gpclarity.interpret_lengthscale` and
:func:`~gpclarity.interpret_variance` for consistent categorisation across
all calls in your analysis.
