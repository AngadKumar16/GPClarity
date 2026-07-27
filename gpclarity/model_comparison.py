"""
Model comparison and selection for Gaussian Processes.

Given several trained GP models of the same data, this module ranks them on a
common footing — marginal likelihood, information criteria (AIC/BIC), structural
complexity, and (optionally) held-out predictive quality — and picks a winner.

All functions consume already-trained models via their ``log_likelihood()`` and
``predict()`` methods; the module itself imports only NumPy, so it loads without
GPy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from gpclarity.exceptions import ComparisonError
from gpclarity.metrics import (
    calibration_error,
    coverage_probability,
    crps_gaussian,
    nlpd_gaussian,
)

logger = logging.getLogger(__name__)

# Criteria where a smaller value is better.
_LOWER_IS_BETTER = {"aic", "bic", "nlpd", "crps", "rmse", "calibration_error"}
# Criteria where a larger value is better.
_HIGHER_IS_BETTER = {"log_likelihood"}
VALID_CRITERIA = _LOWER_IS_BETTER | _HIGHER_IS_BETTER


def _n_free_params(model: Any) -> int:
    """Count the free (optimized) parameters of a GP model.

    Args:
        model: A trained GP model.

    Returns:
        Number of free parameters, preferring GPy's ``optimizer_array`` and
        falling back to ``param_array`` or the ``parameters`` collection.
    """
    for attr in ("optimizer_array", "param_array"):
        arr = getattr(model, attr, None)
        if arr is not None:
            return int(np.asarray(arr).ravel().size)
    if hasattr(model, "parameters"):
        return int(sum(np.asarray(p.param_array).size for p in model.parameters))
    return 0


def _n_data(model: Any) -> int:
    """Return the number of training points backing a model (``0`` if unknown)."""
    X = getattr(model, "X", None)
    if X is not None:
        return int(np.asarray(X).shape[0])
    return 0


def _log_likelihood(model: Any) -> float:
    """Return a model's marginal log-likelihood as a float.

    Args:
        model: A trained GP model exposing ``log_likelihood()``.

    Returns:
        The marginal log-likelihood.

    Raises:
        ComparisonError: If the value cannot be computed.
    """
    try:
        return float(model.log_likelihood())
    except Exception as e:  # noqa: BLE001
        raise ComparisonError(f"could not compute log-likelihood: {e}") from e


@dataclass
class ModelScore:
    """Scores for a single model in a comparison.

    Attributes:
        name: Label of the model.
        log_likelihood: Marginal log-likelihood (higher is better).
        n_params: Number of free parameters.
        n_data: Number of training points.
        aic: Akaike Information Criterion (lower is better).
        bic: Bayesian Information Criterion (lower is better).
        test_metrics: Held-out metrics if test data was supplied, else empty.
    """

    name: str
    log_likelihood: float
    n_params: int
    n_data: int
    aic: float
    bic: float
    test_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return the score row as a plain dictionary (test metrics flattened)."""
        base = {
            "name": self.name,
            "log_likelihood": self.log_likelihood,
            "n_params": self.n_params,
            "n_data": self.n_data,
            "aic": self.aic,
            "bic": self.bic,
        }
        base.update(self.test_metrics)
        return base


@dataclass
class ModelComparison:
    """Result of comparing several models under a chosen criterion.

    Attributes:
        scores: Per-model :class:`ModelScore` rows.
        criterion: The criterion used to rank models.
        ranking: Model names ordered best → worst.
        best: Name of the winning model.
    """

    scores: List[ModelScore]
    criterion: str
    ranking: List[str]
    best: str

    def to_dict(self) -> Dict[str, Any]:
        """Return the comparison as a plain, serialisable dictionary."""
        return {
            "criterion": self.criterion,
            "best": self.best,
            "ranking": self.ranking,
            "scores": [s.to_dict() for s in self.scores],
        }

    def to_table(self) -> str:
        """Render the comparison as a fixed-width text table (best first)."""
        rows = {s.name: s for s in self.scores}
        ordered = [rows[n] for n in self.ranking]
        cols = ["name", "log_likelihood", "n_params", "aic", "bic"]
        extra = sorted({k for s in self.scores for k in s.test_metrics})
        header = cols + extra
        widths = {c: max(len(c), 12) for c in header}
        lines = ["  ".join(c.rjust(widths[c]) for c in header)]
        for s in ordered:
            d = s.to_dict()
            cells = []
            for c in header:
                v = d.get(c, "")
                cells.append(
                    (f"{v:.4g}" if isinstance(v, float) else str(v)).rjust(widths[c])
                )
            lines.append("  ".join(cells))
        return "\n".join(lines)

    def to_dataframe(self):
        """Return the scores as a pandas DataFrame indexed by rank.

        Raises:
            ComparisonError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ComparisonError("to_dataframe requires pandas") from e
        df = pd.DataFrame([s.to_dict() for s in self.scores]).set_index("name")
        return df.loc[self.ranking]


def score_model(
    name: str,
    model: Any,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    confidence_level: float = 1.96,
) -> ModelScore:
    """Compute likelihood, information criteria, and optional test metrics.

    Args:
        name: Label for the model.
        model: A trained GP model with ``log_likelihood()`` and ``predict()``.
        X_test: Optional held-out inputs for predictive metrics.
        y_test: Optional held-out targets (required if ``X_test`` given).
        confidence_level: Interval half-width (in std) for the coverage metric.

    Returns:
        A populated :class:`ModelScore`.

    Raises:
        ComparisonError: If likelihood fails, or test data is inconsistent.
    """
    ll = _log_likelihood(model)
    k = _n_free_params(model)
    n = _n_data(model)
    aic = 2.0 * k - 2.0 * ll
    bic = (k * math.log(n) if n > 0 else 0.0) - 2.0 * ll

    test_metrics: Dict[str, float] = {}
    if X_test is not None:
        if y_test is None:
            raise ComparisonError("y_test is required when X_test is provided")
        try:
            mean, var = model.predict(np.asarray(X_test, dtype=float))
        except Exception as e:  # noqa: BLE001
            raise ComparisonError(f"{name}: predict failed: {e}") from e
        mean = np.asarray(mean).ravel()
        var = np.asarray(var).ravel()
        yt = np.asarray(y_test, dtype=float).ravel()
        test_metrics = {
            "nlpd": nlpd_gaussian(yt, mean, var),
            "crps": crps_gaussian(yt, mean, var),
            "rmse": float(np.sqrt(np.mean((yt - mean) ** 2))),
            "coverage": coverage_probability(yt, mean, var, confidence_level),
            "calibration_error": calibration_error(yt, mean, var),
        }

    return ModelScore(
        name=name,
        log_likelihood=ll,
        n_params=k,
        n_data=n,
        aic=aic,
        bic=bic,
        test_metrics=test_metrics,
    )


def compare_models(
    models: Mapping[str, Any],
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    criterion: str = "bic",
    confidence_level: float = 1.96,
) -> ModelComparison:
    """Score and rank several trained models under a single criterion.

    Args:
        models: Mapping ``{name: trained_model}``. Two or more entries.
        X_test: Optional held-out inputs; enables test-set criteria.
        y_test: Optional held-out targets (required if ``X_test`` given).
        criterion: Ranking key — one of ``"bic"``, ``"aic"``,
            ``"log_likelihood"``, or (with test data) ``"nlpd"``, ``"crps"``,
            ``"rmse"``, ``"calibration_error"``.
        confidence_level: Interval half-width (in std) for coverage.

    Returns:
        A :class:`ModelComparison` with per-model scores and the ranking.

    Raises:
        ComparisonError: If fewer than two models are given, the criterion is
            unknown, or a test-set criterion is requested without test data.

    Example:
        >>> cmp = gpclarity.compare_models({"rbf": m1, "matern": m2}, criterion="bic")
        >>> print(cmp.best)
        >>> print(cmp.to_table())
    """
    criterion = criterion.lower()
    if criterion not in VALID_CRITERIA:
        raise ComparisonError(
            f"unknown criterion {criterion!r}; valid: {sorted(VALID_CRITERIA)}"
        )
    if len(models) < 2:
        raise ComparisonError("need at least two models to compare")

    needs_test = criterion in {"nlpd", "crps", "rmse", "calibration_error"}
    if needs_test and X_test is None:
        raise ComparisonError(f"criterion {criterion!r} requires X_test and y_test")

    scores = [
        score_model(name, model, X_test, y_test, confidence_level)
        for name, model in models.items()
    ]

    def _key(s: ModelScore) -> float:
        if criterion in s.to_dict():
            return float(s.to_dict()[criterion])
        raise ComparisonError(f"criterion {criterion!r} not available in scores")

    reverse = criterion in _HIGHER_IS_BETTER
    ordered = sorted(scores, key=_key, reverse=reverse)
    ranking = [s.name for s in ordered]

    return ModelComparison(
        scores=scores,
        criterion=criterion,
        ranking=ranking,
        best=ranking[0],
    )


def select_best_model(
    models: Mapping[str, Any],
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    criterion: str = "bic",
    confidence_level: float = 1.96,
) -> Tuple[str, Any]:
    """Return the ``(name, model)`` of the best model under ``criterion``.

    Thin convenience wrapper over :func:`compare_models`.

    Args:
        models: Mapping ``{name: trained_model}``.
        X_test: Optional held-out inputs.
        y_test: Optional held-out targets.
        criterion: Ranking key (see :func:`compare_models`).
        confidence_level: Interval half-width (in std) for coverage.

    Returns:
        Tuple ``(best_name, best_model)``.

    Raises:
        ComparisonError: Propagated from :func:`compare_models`.
    """
    result = compare_models(models, X_test, y_test, criterion, confidence_level)
    return result.best, models[result.best]
