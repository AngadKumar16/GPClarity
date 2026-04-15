Installation
============

Requirements
------------

GPClarity requires Python 3.9 or later. The core library has minimal dependencies, but full functionality requires GPy.

Basic Installation
------------------

For analysis and lightweight usage:

.. code-block:: bash

   pip install gpclarity

This installs the core interpretability tools without heavy scientific computing dependencies.

Development Installation
------------------------

To install from source with development tools:

.. code-block:: bash

   git clone https://github.com/AngadKumar16/gpclarity.git
   cd gpclarity
   pip install -e ".[dev]"

Dependencies
------------

Core dependencies:

- numpy >= 1.20
- scipy >= 1.7
- GPy >= 1.10

Optional dependencies (install separately as needed):

- matplotlib >= 3.4 — required for all plot methods
- pandas >= 1.3 — required for ``HyperparameterTracker.to_dataframe()``
- joblib — enables parallel LOO computation in ``DataInfluenceMap``
- tqdm — shows progress bars during LOO computation
- scikit-learn — used for nearest-neighbour extrapolation detection

Verifying Installation
----------------------

.. code-block:: python

   import gpclarity
   print(gpclarity.__version__)
   print(f"Full features available: {gpclarity.AVAILABLE['full']}")