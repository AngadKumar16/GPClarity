"""
Shared numerical utilities and model health checks for GPClarity.
"""

import logging
from typing import Any, Dict, Union

import numpy as np

logger = logging.getLogger(__name__)


class LinAlgError(Exception):
    """Linear algebra computation error."""

    pass


def _validate_kernel_matrix(K: np.ndarray) -> None:
    """
    Validate kernel matrix for numerical issues.

    Args:
        K: Kernel matrix to validate

    Raises:
        LinAlgError: If matrix is invalid
    """
    if not np.all(np.isfinite(K)):
        n_nonfinite = np.sum(~np.isfinite(K))
        raise LinAlgError(f"Kernel matrix contains {n_nonfinite} non-finite values")

    if K.shape[0] != K.shape[1]:
        raise LinAlgError(f"Kernel matrix must be square, got {K.shape}")

    # Check symmetry
    if not np.allclose(K, K.T, rtol=1e-5, atol=1e-8):
        max_asym = np.max(np.abs(K - K.T))
        logger.warning(f"Kernel matrix asymmetric (max diff: {max_asym:.2e})")


def _cholesky_with_jitter(
    K: np.ndarray,
    max_attempts: int = 5,
    initial_jitter: float = 1e-6,
    jitter_growth: float = 10.0,
) -> np.ndarray:
    """
    Compute Cholesky decomposition with progressive jitter.

    Args:
        K: Positive semi-definite matrix
        max_attempts: Maximum jitter attempts
        initial_jitter: Starting jitter magnitude
        jitter_growth: Multiplicative factor for jitter increase

    Returns:
        Lower triangular Cholesky factor

    Raises:
        LinAlgError: If decomposition fails after all attempts
    """
    try:
        return np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        pass

    K_work = K.copy()
    jitter = initial_jitter

    for attempt in range(max_attempts):
        K_work = K_work + np.eye(K.shape[0]) * jitter
        try:
            L = np.linalg.cholesky(K_work)
            logger.debug(f"Cholesky succeeded with jitter {jitter:.2e}")
            return L
        except np.linalg.LinAlgError:
            jitter *= jitter_growth

    raise LinAlgError(
        f"Cholesky decomposition failed after {max_attempts} attempts "
        f"with max jitter {jitter/jitter_growth:.2e}"
    )


def _extract_param_value(param: Any) -> Union[float, np.ndarray]:
    """
    Safely extract scalar or array value from GPy parameter.

    Args:
        param: GPy parameter object

    Returns:
        Scalar float or numpy array
    """
    val = param.param_array

    if val is None:
        return 0.0

    arr = np.atleast_1d(val)

    if len(arr) == 1:
        return float(arr[0])
    else:
        return arr.copy()


def _validate_convergence_window(window: int, history_length: int) -> None:
    """
    Validate window size for convergence analysis.

    Args:
        window: Requested window size
        history_length: Available history length

    Raises:
        ValueError: If window invalid
    """
    if window <= 0:
        raise ValueError(f"Window must be positive, got {window}")
    if window > history_length // 2:
        raise ValueError(
            f"Window ({window}) too large for history length ({history_length}). "
            f"Max allowed: {history_length // 2}"
        )


def _validate_array(arr: Any, name: str = "array") -> np.ndarray:
    """
    Validate and convert input to numpy array.

    Args:
        arr: Input array-like
        name: Name for error messages

    Returns:
        Validated numpy array

    Raises:
        ValueError: If invalid
    """
    if arr is None:
        raise ValueError(f"{name} cannot be None")

    try:
        arr = np.asarray(arr)
    except Exception as e:
        raise ValueError(f"{name} must be array-like: {e}") from e

    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")

    if not np.all(np.isfinite(arr)):
        n_invalid = np.sum(~np.isfinite(arr))
        raise ValueError(f"{name} contains {n_invalid} non-finite values")

    return arr


def check_model_health(model: Any) -> Dict[str, Any]:
    """
    Check if a GP model is healthy and ready for analysis.

    Validates model attributes, checks for non-finite parameter values,
    and evaluates log-likelihood.

    Args:
        model: Trained GP model to inspect

    Returns:
        Dictionary with keys:

        - ``healthy`` (bool): True if no issues found
        - ``issues`` (List[str]): Critical problems that prevent analysis
        - ``warnings`` (List[str]): Non-fatal concerns
        - ``log_likelihood`` (Optional[float]): Model log-likelihood if computable
        - ``n_parameters`` (int): Number of model parameters
    """
    issues = []
    warnings_list = []

    # Check basic attributes
    if not hasattr(model, "predict"):
        issues.append("Model missing predict() method")
    if not hasattr(model, "kern"):
        issues.append("Model missing kern attribute")

    # Check parameters
    if hasattr(model, "parameters"):
        for param in model.parameters:
            if hasattr(param, "param_array"):
                arr = param.param_array
                if not np.all(np.isfinite(arr)):
                    issues.append(f"Parameter {param.name} has non-finite values")

    # Check log-likelihood
    ll = None
    if hasattr(model, "log_likelihood") and not issues:
        try:
            ll = float(model.log_likelihood())
            if not np.isfinite(ll):
                warnings_list.append("Log-likelihood is not finite")
            elif ll > 0:
                warnings_list.append(
                    "Log-likelihood is positive (unusual for regression)"
                )
        except Exception as e:
            warnings_list.append(f"Could not compute log-likelihood: {e}")

    return {
        "healthy": len(issues) == 0,
        "issues": issues,
        "warnings": warnings_list,
        "log_likelihood": ll,
        "n_parameters": len(model.parameters) if hasattr(model, "parameters") else 0,
    }
