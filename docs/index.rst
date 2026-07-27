GPClarity: Interpretability Toolkit for Gaussian Processes
==========================================================

.. image:: https://img.shields.io/pypi/v/gpclarity.svg
   :target: https://pypi.org/project/gpclarity/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.9%2B-blue.svg
   :alt: Python 3.9+

**Make your Gaussian Process models transparent.**

GPClarity provides human-readable diagnostics, uncertainty quantification, and model
introspection tools that bridge the gap between black-box GP predictions and actionable
insights.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference/index

Features
--------

* **Kernel Interpretation**: Translate cryptic hyperparameters into natural language insights
* **Uncertainty Quantification**: Go beyond variance — detect extrapolation, miscalibration,
  and epistemic vs. aleatoric uncertainty
* **Complexity Analysis**: Measure model capacity, detect overfitting/underfitting,
  get actionable recommendations
* **Optimization Tracking**: Monitor hyperparameter convergence with publication-ready
  visualizations
* **Data Influence**: Identify high-leverage points and quantify leave-one-out effects
* **Predictive Metrics**: Score calibration and sharpness (NLPD, CRPS, coverage, PIT) and cross-validate
* **Kernel Recommendation**: Get a data-driven kernel suggestion with a plain-language rationale
* **Model Comparison**: Rank candidate models by AIC/BIC, likelihood, or held-out metrics
* **Diagnostic Reports**: Produce a one-call Markdown/HTML/JSON report of every diagnostic

Why GPClarity?
--------------

Gaussian Processes are powerful but opaque. GPClarity answers questions like:

- *"Is my kernel too simple or too complex?"*
- *"Where can I trust my model's predictions?"*
- *"Which data points drive my model's behavior?"*
- *"Has my optimization converged or is it stuck?"*

Installation
------------

**Basic** (analysis only):

.. code-block:: bash

   pip install gpclarity

**Full** (with GPy/emukit):

.. code-block:: bash

   pip install gpclarity[full]

**Development**:

.. code-block:: bash

   pip install gpclarity[dev]

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
