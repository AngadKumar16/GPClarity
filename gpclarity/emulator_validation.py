"""
Emulator validation diagnostics for Gaussian Process models.

Implements the validation diagnostics of Bastos & O'Hagan (2009),
"Diagnostics for Gaussian Process Emulators", *Technometrics* 51(4), 425-438
(doi:10.1198/TECH.2009.08019). They compare a GP's joint predictive
distribution against held-out validation data:

* **Individual standardized errors** — each validation residual divided by its
  predictive standard deviation. Under a valid emulator these are roughly
  standard normal.
* **Mahalanobis distance** — a single summary of all residuals that accounts for
  their predictive correlation. Under a valid emulator it follows (approximately)
  a chi-squared distribution with one degree of freedom per validation point.
* **Pivoted Cholesky errors** — the residuals decorrelated through a pivoted
  Cholesky factor of the predictive covariance. The pivoting orders points from
  the most to the least uncertain, which helps separate a misjudged *variance*
  (large errors early in the order) from a misjudged *correlation structure*,
  e.g. lengthscales that are too long (large errors late in the order).

:func:`emulator_diagnostics` is pure NumPy/SciPy and works on any predictive
mean and covariance. :func:`validate_emulator` is a convenience wrapper for GPy
models.

Note:
    The chi-squared reference treats the GP hyperparameters as known. When they
    are estimated, Bastos & O'Hagan show the scaled Mahalanobis distance follows
    an F distribution instead; with moderately sized training sets the
    difference is small, but borderline p-values should be read with care.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
from scipy.linalg import lapack, solve_triangular

from gpclarity.exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class EmulatorValidation:
    """
    Result of a Bastos & O'Hagan emulator validation.

    Attributes:
        n_points: Number of validation points.
        standardized_errors: Residuals divided by predictive standard deviation.
        mahalanobis: Mahalanobis distance of the validation residuals.
        mahalanobis_expected: Expected value under a valid emulator.
        mahalanobis_std: Standard deviation under a valid emulator.
        mahalanobis_pvalue: Two-sided p-value against the chi-squared reference.
        pivoted_cholesky_errors: Decorrelated residuals in pivot order.
        pivot_order: Validation-point indices in pivot order (most uncertain first).
        coverage_95: Fraction of points with ``|standardized error| <= 1.96``.
        n_large_errors: Number of points with ``|standardized error| > 2``.
        verdict: ``"consistent"``, ``"overconfident"`` or ``"underconfident"``.
        interpretation: Plain-language reading of the diagnostics.
    """

    n_points: int
    standardized_errors: np.ndarray
    mahalanobis: float
    mahalanobis_expected: float
    mahalanobis_std: float
    mahalanobis_pvalue: float
    pivoted_cholesky_errors: np.ndarray
    pivot_order: np.ndarray
    coverage_95: float
    n_large_errors: int
    verdict: str
    interpretation: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary of the results."""
        return {
            "n_points": self.n_points,
            "standardized_errors": self.standardized_errors.tolist(),
            "mahalanobis": self.mahalanobis,
            "mahalanobis_expected": self.mahalanobis_expected,
            "mahalanobis_std": self.mahalanobis_std,
            "mahalanobis_pvalue": self.mahalanobis_pvalue,
            "pivoted_cholesky_errors": self.pivoted_cholesky_errors.tolist(),
            "pivot_order": self.pivot_order.tolist(),
            "coverage_95": self.coverage_95,
            "n_large_errors": self.n_large_errors,
            "verdict": self.verdict,
            "interpretation": list(self.interpretation),
        }


def _validate_inputs(
    y_true: Any, mean: Any, cov: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coerce and check a validation triple.

    Args:
        y_true: Observed validation outputs, shape ``(m,)`` or ``(m, 1)``.
        mean: Predictive mean, same length as ``y_true``.
        cov: Predictive covariance, shape ``(m, m)``.

    Returns:
        Tuple ``(y_true, mean, cov)`` as float arrays of shapes ``(m,)``,
        ``(m,)`` and ``(m, m)``.

    Raises:
        ValidationError: On shape mismatch, non-finite values, or a covariance
            that is not symmetric with a positive diagonal.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    mu = np.asarray(mean, dtype=float).ravel()
    V = np.asarray(cov, dtype=float)

    if y.size < 2:
        raise ValidationError("need at least 2 validation points")
    if mu.shape != y.shape:
        raise ValidationError(f"shape mismatch: y_true{y.shape}, mean{mu.shape}")
    if V.shape != (y.size, y.size):
        raise ValidationError(f"covariance must be ({y.size}, {y.size}), got {V.shape}")
    for name, arr in (("y_true", y), ("mean", mu), ("cov", V)):
        if not np.all(np.isfinite(arr)):
            raise ValidationError(f"{name} contains non-finite values")
    if not np.allclose(V, V.T, rtol=1e-6, atol=1e-10):
        raise ValidationError("covariance is not symmetric")
    if np.any(np.diag(V) <= 0):
        raise ValidationError("covariance has non-positive diagonal entries")
    return y, mu, 0.5 * (V + V.T)


def _pivoted_cholesky(V: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Pivoted Cholesky factorisation ``V[piv][:, piv] = G @ G.T``.

    Args:
        V: Symmetric positive semi-definite matrix.

    Returns:
        Tuple ``(G, piv, rank)`` where ``G`` is lower triangular, ``piv`` holds
        0-based pivot indices and ``rank`` is the numerical rank.

    Raises:
        ValidationError: If LAPACK reports an error.
    """
    c, piv, rank, info = lapack.dpstrf(V, lower=1)
    if info < 0:
        raise ValidationError(f"pivoted Cholesky failed (LAPACK info={info})")
    G = np.tril(c)
    return G, piv - 1, int(rank)


def _interpret(
    verdict: str,
    pvalue: float,
    n_large: int,
    n: int,
    pc_errors: np.ndarray,
    threshold: float,
) -> List[str]:
    """Build plain-language notes from the diagnostic values."""
    notes: List[str] = []
    if verdict == "consistent":
        notes.append(
            f"Mahalanobis distance is consistent with the emulator "
            f"(p={pvalue:.3f}); no evidence of miscalibration overall."
        )
    elif verdict == "overconfident":
        notes.append(
            f"Mahalanobis distance is too large (p={pvalue:.3g}): predictive "
            "uncertainty is too small for the observed errors."
        )
    else:
        notes.append(
            f"Mahalanobis distance is too small (p={pvalue:.3g}): predictive "
            "uncertainty is larger than the observed errors justify."
        )

    expected_large = 0.0455 * n
    if n_large > max(1.0, 2.0 * expected_large):
        notes.append(
            f"{n_large}/{n} standardized errors exceed 2 (about "
            f"{expected_large:.1f} expected); check whether they cluster in one "
            "region of input space, which points to non-stationarity."
        )

    # Heuristic reading of the pivoted Cholesky errors. Bastos & O'Hagan inspect
    # these graphically; the early/late split below only flags clear patterns.
    large = np.abs(pc_errors) > threshold
    if large.sum() >= 2:
        half = len(pc_errors) // 2
        early = float(np.mean(pc_errors[:half] ** 2))
        late = float(np.mean(pc_errors[half:] ** 2))
        if late > 3 * early:
            notes.append(
                "Large pivoted Cholesky errors concentrate late in the pivot "
                "order, which may indicate a misspecified correlation structure "
                "(for example, lengthscales that are too long)."
            )
        elif early > 3 * late:
            notes.append(
                "Large pivoted Cholesky errors concentrate early in the pivot "
                "order, which may indicate the variance at the most uncertain "
                "points is underestimated."
            )
        else:
            notes.append(
                "Large pivoted Cholesky errors are spread across the pivot "
                "order, which may indicate the overall signal variance is "
                "underestimated. Plot the errors against pivot index to check."
            )
    return notes


def emulator_diagnostics(
    y_true: Any,
    mean: Any,
    cov: Any,
    alpha: float = 0.05,
    error_threshold: float = 2.0,
) -> EmulatorValidation:
    """
    Compute Bastos & O'Hagan validation diagnostics from a joint prediction.

    Args:
        y_true: Observed outputs at the validation inputs.
        mean: Predictive mean at the validation inputs.
        cov: Full predictive covariance at the validation inputs. Include the
            observation noise when the validation outputs are noisy.
        alpha: Significance level for the Mahalanobis verdict.
        error_threshold: Absolute value above which a pivoted Cholesky error
            counts as large in the interpretation.

    Returns:
        An :class:`EmulatorValidation` with all diagnostics and a verdict.

    Raises:
        ValidationError: If inputs are malformed.
    """
    if not 0.0 < alpha < 1.0:
        raise ValidationError(f"alpha must be in (0, 1), got {alpha}")

    y, mu, V = _validate_inputs(y_true, mean, cov)
    n = y.size
    resid = y - mu

    std_errors = resid / np.sqrt(np.diag(V))

    G, piv, rank = _pivoted_cholesky(V)
    if rank < n:
        logger.warning(
            "Predictive covariance is rank-deficient (rank %d of %d); "
            "using the leading %d pivoted components.",
            rank,
            n,
            rank,
        )
    pc_errors = solve_triangular(G[:rank, :rank], resid[piv[:rank]], lower=True)

    # D^2 = r^T V^{-1} r equals the squared norm of the pivoted Cholesky errors.
    mahal = float(np.sum(pc_errors**2))
    dof = rank
    expected = float(dof)
    std = float(np.sqrt(2.0 * dof))
    cdf = float(stats.chi2.cdf(mahal, dof))
    pvalue = float(min(1.0, 2.0 * min(cdf, 1.0 - cdf)))

    if pvalue >= alpha:
        verdict = "consistent"
    elif mahal > expected:
        verdict = "overconfident"
    else:
        verdict = "underconfident"

    n_large = int(np.sum(np.abs(std_errors) > 2.0))
    coverage = float(np.mean(np.abs(std_errors) <= 1.96))

    return EmulatorValidation(
        n_points=n,
        standardized_errors=std_errors,
        mahalanobis=mahal,
        mahalanobis_expected=expected,
        mahalanobis_std=std,
        mahalanobis_pvalue=pvalue,
        pivoted_cholesky_errors=pc_errors,
        pivot_order=piv[:rank],
        coverage_95=coverage,
        n_large_errors=n_large,
        verdict=verdict,
        interpretation=_interpret(
            verdict, pvalue, n_large, n, pc_errors, error_threshold
        ),
    )


def validate_emulator(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    include_likelihood: bool = True,
    alpha: float = 0.05,
) -> EmulatorValidation:
    """
    Validate a fitted GPy model against held-out data (Bastos & O'Hagan 2009).

    Args:
        model: Fitted GPy model exposing ``predict(X, full_cov=True)``.
        X_val: Validation inputs, shape ``(m, d)``. Must not overlap the
            training data.
        y_val: Validation outputs, shape ``(m,)`` or ``(m, 1)``.
        include_likelihood: Add observation noise to the predictive
            covariance. Keep ``True`` for noisy outputs; set ``False`` for
            deterministic simulators.
        alpha: Significance level for the Mahalanobis verdict.

    Returns:
        An :class:`EmulatorValidation`.

    Raises:
        ValidationError: If the model cannot produce a joint prediction or the
            inputs are malformed.

    Example:
        >>> result = validate_emulator(model, X_val, y_val)
        >>> result.verdict
        'consistent'
    """
    if not hasattr(model, "predict"):
        raise ValidationError("model must provide a predict() method")

    X_val = np.asarray(X_val, dtype=float)
    if X_val.ndim == 1:
        X_val = X_val.reshape(-1, 1)

    try:
        mean, cov = model.predict(
            X_val, full_cov=True, include_likelihood=include_likelihood
        )
    except Exception as e:
        raise ValidationError(f"joint prediction failed: {e}") from e

    return emulator_diagnostics(y_val, mean, cov, alpha=alpha)
