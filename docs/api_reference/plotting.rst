Plotting
========

The ``plotting`` module is the rendering backend used by all GPClarity classes.
Each of the three functions is normally invoked via the class method (e.g.
``profiler.plot()``, ``tracker.plot_evolution()``, ``influence.plot_influence()``),
which set sensible defaults and forward to these functions. Call them directly
only when you need full control over figure/axes layout.

.. note::
   Prefer calling ``profiler.plot(X_test)``, ``tracker.plot_evolution()``, and
   ``influence.plot_influence(X_train)`` on the class objects rather than
   importing and calling these functions directly.

.. code-block:: python

   import matplotlib.pyplot as plt
   from gpclarity.plotting import plot_uncertainty_profile

   fig, ax = plt.subplots()
   plot_uncertainty_profile(
       model, X_test, X_train=X_train, y_train=y_train,
       confidence_level=2.0, show_regions=True, ax=ax
   )
   plt.tight_layout()
   plt.show()

.. automodule:: gpclarity.plotting
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autosummary::
   :nosignatures:

   plot_influence_map
   plot_optimization_trajectory
   plot_uncertainty_profile
