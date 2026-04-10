# Contributing to GPClarity

Thank you for considering a contribution! This document covers how to get set up, the conventions we follow, and how to submit changes.

---

## Development Setup

```bash
git clone https://github.com/AngadKumar16/gpclarity.git
cd gpclarity
pip install -e ".[dev]"
```

The `dev` extra installs: `pytest`, `pytest-cov`, `black`, `isort`, `mypy`, `pre-commit`.

Enable pre-commit hooks (runs black and isort automatically on commit):

```bash
pre-commit install
```

---

## Running Tests

```bash
# All tests with coverage report
pytest tests/ -v --cov=gpclarity --cov-report=html

# Single module
pytest tests/test_uncertainty_analysis.py -v

# Quick smoke check
pytest tests/ -q
```

All tests must pass before submitting a PR.

---

## Code Style

- **Formatter:** `black` with `line-length = 88`
- **Import order:** `isort` (black-compatible profile)
- **Type hints:** Required on all public function signatures

Run both manually:

```bash
black gpclarity/ tests/
isort gpclarity/ tests/
```

---

## Docstrings

All public functions and classes require Google-style docstrings with `Args`, `Returns`, and `Raises` sections. Private helpers (`_name`) need at minimum a one-line summary.

**Template:**

```python
def my_function(model: Any, X: np.ndarray) -> Dict[str, Any]:
    """
    One-line summary of what this does.

    Longer explanation if the behavior is non-obvious. Include
    any important algorithmic notes here.

    Args:
        model: Trained GP model with kern and predict attributes.
        X: Training input locations of shape (n_samples, n_features).

    Returns:
        Dictionary with keys:

        - ``score`` (float): The computed score.
        - ``interpretation`` (str): Human-readable category.

    Raises:
        ValueError: If X is empty or contains non-finite values.
        ComplexityError: If the underlying computation fails.

    Example:
        >>> result = my_function(model, X)
        >>> print(result["score"])
    """
```

---

## Project Structure

```
gpclarity/
├── kernel_summary.py      # Kernel parsing and interpretation
├── uncertainty_analysis.py # Uncertainty profiling and calibration
├── hyperparam_tracker.py  # Optimization tracking
├── model_complexity.py    # Complexity scoring (3 strategies)
├── data_influence.py      # Leverage scores and LOO analysis
├── plotting.py            # Rendering backend for all plot methods
├── utils.py               # Shared numerical utilities + check_model_health
└── exceptions.py          # Custom exception hierarchy

tests/
├── conftest.py            # Shared fixtures (simple_gp, composite_gp, X_test)
└── test_*.py              # One file per module

docs/
├── conf.py                # Sphinx configuration (RTD theme, Napoleon)
├── quickstart.rst         # 5-minute tour
├── user_guide/            # Narrative guides per module
└── api_reference/         # Auto-generated API reference via autodoc
```

---

## Adding a New Feature

1. Implement in the appropriate module (or create a new one if truly separate)
2. Export from `gpclarity/__init__.py` and add to `__all__`
3. Write tests in `tests/test_<module>.py` using the shared fixtures in `conftest.py`
4. Document with a Google-style docstring
5. Add the function to the relevant `.rst` files under `docs/`
6. Add an entry to `CHANGELOG.md` under an `[Unreleased]` section

---

## Submitting a Pull Request

1. Fork the repo and create a branch: `git checkout -b feature/my-feature`
2. Make your changes and ensure tests pass
3. Push and open a PR against `main`
4. In the PR description, explain **what** changed and **why**
5. Reference any related issues: `Closes #42`

**Branch naming:**
- `feature/<name>` — new functionality
- `fix/<name>` — bug fixes
- `docs/<name>` — documentation only
- `refactor/<name>` — code quality without behaviour change

---

## Reporting Bugs

Open an issue at [GitHub Issues](https://github.com/AngadKumar16/gpclarity/issues) with:

- A minimal reproducible example
- The full traceback
- Your Python and GPy versions (`python --version`, `pip show GPy`)

---

## Questions

Open a [GitHub Discussion](https://github.com/AngadKumar16/gpclarity/discussions) for questions that aren't bugs or feature requests.
