"""
Predictive metrics and calibration diagnostics for Gaussian Process models.

This module scores the quality of probabilistic predictions rather than the
model's internal structure. The scoring functions are pure NumPy and operate on
predicted means/variances and observed targets, so they work without GPy. The
:func:`cross_validate` helper retrains a model on held-out folds and therefore
requires a working GPy install at call time.

Two families of tools live here:

* **Proper scoring rules** — :func:`nlpd_gaussian`, :func:`crps_gaussian`,
  :func:`interval_score`. Lower is better; they reward calibrated *and* sharp
  predictions.
* **Calibration diagnostics** — :func:`coverage_probability`, :func:`pit_values`,
  :func:`calibration_curve`, :func:`calibration_error`, :func:`sharpness`. These
  answer "are the stated confidence intervals honest?".
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from gpclarity.exceptions import MetricError

logger = logging.getLogger(__name__)

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _as_1d(name: str, arr: Any) -> np.ndarray:
    """Coerce an array-like to a finite 1-D float array.

    Args:
        name: Label used in error messages.
        arr: Array-like input (list, column vector, or 1-D array).

    Returns:
        A contiguous 1-D ``float`` array.

    Raises:
        MetricError: If the input is empty or contains non-finite values.
    """
    a = np.asarray(arr, dtype=float).ravel()
    if a.size == 0:
        raise MetricError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise MetricError(f"{name} contains non-finite values")
    return a


def _align(
    y_true: Any, mean: Any, variance: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and broadcast prediction triples to matching 1-D arrays.

    Args:
        y_true: Observed target values.
        mean: Predictive mean per point.
        variance: Predictive variance per point (must be positive).

    Returns:
        Tuple ``(y_true, mean, std)`` as aligned 1-D arrays, where ``std`` is the
        square root of the (floored) variance.

    Raises:
        MetricError: If lengths differ or variance is non-positive.
    """
    yt = _as_1d("y_true", y_true)
    mu = _as_1d("mean", mean)
    var = _as_1d("variance", variance)
    if not (yt.shape == mu.shape == var.shape):
        raise MetricError(
            f"shape mismatch: y_true{yt.shape}, mean{mu.shape}, variance{var.shape}"
        )
    if np.any(var <= 0):
        raise MetricError("variance must be strictly positive")
    return yt, mu, np.sqrt(var)


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    """Standard-normal probability density."""
    return np.exp(-0.5 * z * z) / _SQRT2PI


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """Standard-normal cumulative distribution via ``erf``."""
    from math import erf

    vec = np.vectorize(lambda t: 0.5 * (1.0 + erf(t / _SQRT2)))
    return vec(z)


def nlpd_gaussian(y_true: Any, mean: Any, variance: Any) -> float:
    """Mean negative log predictive density under a Gaussian likelihood.

    Also called the negative log-likelihood of held-out data. It is a strictly
    proper scoring rule: it is minimized only when both the mean and the variance
    are correct. Lower is better.

    Args:
        y_true: Observed targets, shape ``(n,)`` or ``(n, 1)``.
        mean: Predictive mean per point.
        variance: Predictive variance per point (strictly positive).

    Returns:
        Mean NLPD across all points (nats).

    Raises:
        MetricError: If inputs are misaligned or variance is non-positive.
    """
    yt, mu, std = _align(y_true, mean, variance)
    var = std * std
    ll = -0.5 * np.log(2.0 * np.pi * var) - 0.5 * (yt - mu) ** 2 / var
    return float(-np.mean(ll))


def crps_gaussian(y_true: Any, mean: Any, variance: Any) -> float:
    """Mean Continuous Ranked Probability Score for Gaussian predictions.

    CRPS has a closed form for a normal predictive distribution
    (Gneiting & Raftery, 2007). It is reported in the units of ``y`` and, unlike
    NLPD, stays finite even for badly placed points, which makes it robust for
    ranking models. Lower is better.

    Args:
        y_true: Observed targets.
        mean: Predictive mean per point.
        variance: Predictive variance per point (strictly positive).

    Returns:
        Mean CRPS across all points, in the units of ``y``.

    Raises:
        MetricError: If inputs are misaligned or variance is non-positive.
    """
    yt, mu, std = _align(y_true, mean, variance)
    z = (yt - mu) / std
    # closed form: sigma * ( z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) )
    crps = std * (
        z * (2.0 * _norm_cdf(z) - 1.0) + 2.0 * _norm_pdf(z) - 1.0 / math.sqrt(math.pi)
    )
    return float(np.mean(crps))


def coverage_probability(
    y_true: Any, mean: Any, variance: Any, confidence_level: float = 2.0
) -> float:
    """Empirical fraction of targets inside ``mean ± confidence_level·std``.

    For well-calibrated Gaussian predictions this should approach the nominal
    coverage of the interval (e.g. ``confidence_level=1.96`` → 0.95).

    Args:
        y_true: Observed targets.
        mean: Predictive mean per point.
        variance: Predictive variance per point.
        confidence_level: Half-width of the interval in standard deviations.

    Returns:
        Observed coverage fraction in ``[0, 1]``.

    Raises:
        MetricError: If inputs are misaligned or ``confidence_level`` ≤ 0.
    """
    if confidence_level <= 0:
        raise MetricError("confidence_level must be positive")
    yt, mu, std = _align(y_true, mean, variance)
    half = confidence_level * std
    inside = np.abs(yt - mu) <= half
    return float(np.mean(inside))


def nominal_coverage(confidence_level: float = 2.0) -> float:
    """Nominal (target) coverage of a symmetric Gaussian interval.

    Args:
        confidence_level: Half-width of the interval in standard deviations.

    Returns:
        The probability mass inside ``±confidence_level`` under a standard normal.
    """
    from math import erf

    return float(erf(confidence_level / _SQRT2))


def pit_values(y_true: Any, mean: Any, variance: Any) -> np.ndarray:
    """Probability Integral Transform values for each prediction.

    PIT value ``= Phi((y - mean) / std)``. If the predictive distribution is
    correct, PIT values are uniformly distributed on ``[0, 1]``; systematic
    deviation reveals miscalibration (a PIT histogram that piles up in the middle
    means the model is under-confident; U-shaped means over-confident).

    Args:
        y_true: Observed targets.
        mean: Predictive mean per point.
        variance: Predictive variance per point.

    Returns:
        Array of PIT values, shape ``(n,)``.

    Raises:
        MetricError: If inputs are misaligned or variance is non-positive.
    """
    yt, mu, std = _align(y_true, mean, variance)
    return _norm_cdf((yt - mu) / std)


def calibration_curve(
    y_true: Any, mean: Any, variance: Any, n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """Reliability curve of expected vs. observed coverage.

    Sweeps a set of nominal confidence levels and, for each, measures the
    empirical fraction of targets whose central credible interval contains them.
    A perfectly calibrated model lies on the diagonal.

    Args:
        y_true: Observed targets.
        mean: Predictive mean per point.
        variance: Predictive variance per point.
        n_bins: Number of nominal confidence levels between 0 and 1 (exclusive).

    Returns:
        Dictionary with keys ``expected`` (nominal probabilities) and
        ``observed`` (empirical coverage), each of length ``n_bins``.

    Raises:
        MetricError: If ``n_bins < 2`` or inputs are invalid.
    """
    if n_bins < 2:
        raise MetricError("n_bins must be >= 2")
    yt, mu, std = _align(y_true, mean, variance)
    pit = _norm_cdf((yt - mu) / std)
    expected = np.linspace(0.0, 1.0, n_bins + 2)[1:-1]
    observed = np.array([np.mean(pit <= p) for p in expected])
    return {"expected": expected, "observed": observed}


def calibration_error(
    y_true: Any, mean: Any, variance: Any, n_bins: int = 10
) -> float:
    """Expected calibration error: mean |observed − expected| coverage gap.

    A single scalar summarising the reliability curve from
    :func:`calibration_curve`; 0 is perfect. Useful for ranking models.

    Args:
        y_true: Observed targets.
        mean: Predictive mean per point.
        variance: Predictive variance per point.
        n_bins: Number of nominal confidence levels.

    Returns:
        Mean absolute gap between observed and expected coverage.
    """
    curve = calibration_curve(y_true, mean, variance, n_bins=n_bins)
    return float(np.mean(np.abs(curve["observed"] - curve["expected"])))


def sharpness(variance: Any) -> float:
    """Average predictive standard deviation (interval width proxy).

    Sharpness measures how tight the predictions are, independent of the targets.
    Read it alongside a calibration metric: the goal is the sharpest model that
    is still calibrated.

    Args:
        variance: Predictive variance per point (strictly positive).

    Returns:
        Root-mean predictive variance, i.e. a representative standard deviation.

    Raises:
        MetricError: If variance is empty or non-positive.
    """
    var = _as_1d("variance", variance)
    if np.any(var <= 0):
        raise MetricError("variance must be strictly positive")
    return float(np.sqrt(np.mean(var)))


def interval_score(
    y_true: Any, mean: Any, variance: Any, alpha: float = 0.05
) -> float:
    """Mean interval score for the central ``(1 − alpha)`` prediction interval.

    The interval score (Gneiting & Raftery, 2007) rewards narrow intervals and
    penalises observations that fall outside them, with the penalty scaled by
    ``2/alpha``. Lower is better.

    Args:
        y_true: Observed targets.
        mean: Predictive mean per point.
        variance: Predictive variance per point.
        alpha: Miscoverage level; the interval has nominal coverage ``1 - alpha``.

    Returns:
        Mean interval score across all points.

    Raises:
        MetricError: If ``alpha`` is outside ``(0, 1)`` or inputs are invalid.
    """
    if not (0.0 < alpha < 1.0):
        raise MetricError("alpha must be in (0, 1)")
    yt, mu, std = _align(y_true, mean, variance)
    # two-sided z for central interval
    from math import erf

    def _ppf(p: float) -> float:
        # invert standard normal CDF via bisection (dependency-free)
        lo, hi = -12.0, 12.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 0.5 * (1.0 + erf(mid / _SQRT2)) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    z = _ppf(1.0 - alpha / 2.0)
    lower = mu - z * std
    upper = mu + z * std
    width = upper - lower
    below = (lower - yt) * (yt < lower)
    above = (yt - upper) * (yt > upper)
    score = width + (2.0 / alpha) * (below + above)
    return float(np.mean(score))


@dataclass
class CVResult:
    """Aggregated cross-validation scores across folds.

    Attributes:
        n_folds: Number of folds evaluated.
        nlpd: Mean NLPD across folds.
        crps: Mean CRPS across folds.
        rmse: Root-mean-squared error of the predictive mean across folds.
        coverage: Mean empirical coverage of the ``confidence_level`` interval.
        calibration_error: Mean expected calibration error across folds.
        per_fold: List of per-fold metric dictionaries.
    """

    n_folds: int
    nlpd: float
    crps: float
    rmse: float
    coverage: float
    calibration_error: float
    per_fold: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return the aggregated scores as a plain dictionary."""
        return {
            "n_folds": self.n_folds,
            "nlpd": self.nlpd,
            "crps": self.crps,
            "rmse": self.rmse,
            "coverage": self.coverage,
            "calibration_error": self.calibration_error,
            "per_fold": self.per_fold,
        }


def cross_validate(
    model_factory: Callable[[np.ndarray, np.ndarray], Any],
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    confidence_level: float = 1.96,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> CVResult:
    """K-fold cross-validate a GP-building function on ``(X, y)``.

    For each fold the ``model_factory`` is called on the training split to build
    and train a fresh model, which is then scored on the held-out split. This
    requires GPy (or whatever backend the factory uses) at call time.

    Args:
        model_factory: Callable ``(X_train, y_train) -> trained_model`` that
            builds *and* optimizes a model. The returned model must expose
            ``predict(X) -> (mean, variance)``.
        X: Inputs, shape ``(n, d)``.
        y: Targets, shape ``(n,)`` or ``(n, 1)``.
        n_folds: Number of folds (``2 <= n_folds <= n``).
        confidence_level: Interval half-width (in std) for the coverage metric.
        shuffle: Whether to shuffle indices before splitting.
        random_state: Seed for the shuffle, for reproducibility.

    Returns:
        A :class:`CVResult` with mean scores and per-fold detail.

    Raises:
        MetricError: If ``n_folds`` is invalid or a fold fails to train/predict.

    Example:
        >>> import GPy, numpy as np, gpclarity
        >>> def factory(Xtr, ytr):
        ...     m = GPy.models.GPRegression(Xtr, ytr, GPy.kern.RBF(Xtr.shape[1]))
        ...     m.optimize()
        ...     return m
        >>> cv = gpclarity.cross_validate(factory, X, y, n_folds=5)
        >>> print(cv.nlpd, cv.coverage)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n = X.shape[0]
    if not (2 <= n_folds <= n):
        raise MetricError(f"n_folds must be in [2, {n}], got {n_folds}")

    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)

    per_fold: List[Dict[str, float]] = []
    for k in range(n_folds):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        try:
            model = model_factory(X[train_idx], y[train_idx])
            mean, var = model.predict(X[test_idx])
        except Exception as e:  # noqa: BLE001 - surface as library error
            raise MetricError(f"fold {k} failed: {e}") from e
        mean = np.asarray(mean).ravel()
        var = np.asarray(var).ravel()
        yt = y[test_idx].ravel()
        per_fold.append(
            {
                "nlpd": nlpd_gaussian(yt, mean, var),
                "crps": crps_gaussian(yt, mean, var),
                "rmse": float(np.sqrt(np.mean((yt - mean) ** 2))),
                "coverage": coverage_probability(yt, mean, var, confidence_level),
                "calibration_error": calibration_error(yt, mean, var),
            }
        )

    def _avg(key: str) -> float:
        return float(np.mean([f[key] for f in per_fold]))

    return CVResult(
        n_folds=n_folds,
        nlpd=_avg("nlpd"),
        crps=_avg("crps"),
        rmse=_avg("rmse"),
        coverage=_avg("coverage"),
        calibration_error=_avg("calibration_error"),
        per_fold=per_fold,
    )


def compute_all_metrics(
    y_true: Any, mean: Any, variance: Any, confidence_level: float = 1.96
) -> Dict[str, float]:
    """Convenience wrapper returning every point-metric in one dict.

    Args:
        y_true: Observed targets.
        mean: Predictive mean per point.
        variance: Predictive variance per point.
        confidence_level: Interval half-width (in std) for coverage.

    Returns:
        Dictionary with ``nlpd``, ``crps``, ``rmse``, ``coverage``,
        ``nominal_coverage``, ``calibration_error``, ``sharpness``, and
        ``interval_score``.
    """
    yt, mu, std = _align(y_true, mean, variance)
    var = std * std
    return {
        "nlpd": nlpd_gaussian(yt, mu, var),
        "crps": crps_gaussian(yt, mu, var),
        "rmse": float(np.sqrt(np.mean((yt - mu) ** 2))),
        "coverage": coverage_probability(yt, mu, var, confidence_level),
        "nominal_coverage": nominal_coverage(confidence_level),
        "calibration_error": calibration_error(yt, mu, var),
        "sharpness": sharpness(var),
        "interval_score": interval_score(yt, mu, var),
    }
