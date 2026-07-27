"""
Data-driven kernel recommendation for Gaussian Process regression.

Choosing a kernel is the hardest part of GP modelling. This module inspects the
raw data — its trend, periodicity, smoothness, and noise — and suggests a kernel
structure with a plain-language rationale, so you start from an informed guess
instead of a blind ``RBF``.

The analysis in :func:`analyze_data_characteristics` and :func:`suggest_kernel`
is pure NumPy and works without GPy. :func:`build_kernel` turns a recommendation
into a concrete ``GPy.kern`` object and therefore needs GPy at call time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gpclarity.exceptions import RecommendationError

logger = logging.getLogger(__name__)


@dataclass
class DataCharacteristics:
    """Summary statistics describing the shape of a 1-D signal.

    Attributes:
        n_points: Number of observations.
        input_range: ``(min, max)`` of the (first) input dimension.
        trend_strength: R² of a linear fit; high means a strong global trend.
        trend_slope: Slope of the linear fit in ``y`` per ``x`` units.
        periodicity_strength: Normalised strength of the dominant FFT peak in
            ``[0, 1]``; high means a clear repeating pattern.
        dominant_period: Estimated period of the strongest cycle, or ``None``.
        smoothness: Roughness proxy in ``[0, 1]`` from lag-1 autocorrelation of
            differences; high means smooth, low means wiggly.
        noise_ratio: Estimated fraction of variance attributable to noise.
        monotonic: Whether the signal is (weakly) monotonic.
    """

    n_points: int
    input_range: Tuple[float, float]
    trend_strength: float
    trend_slope: float
    periodicity_strength: float
    dominant_period: Optional[float]
    smoothness: float
    noise_ratio: float
    monotonic: bool

    def to_dict(self) -> Dict[str, Any]:
        """Return the characteristics as a plain dictionary."""
        return {
            "n_points": self.n_points,
            "input_range": self.input_range,
            "trend_strength": self.trend_strength,
            "trend_slope": self.trend_slope,
            "periodicity_strength": self.periodicity_strength,
            "dominant_period": self.dominant_period,
            "smoothness": self.smoothness,
            "noise_ratio": self.noise_ratio,
            "monotonic": self.monotonic,
        }


@dataclass
class KernelRecommendation:
    """A suggested kernel structure with justification.

    Attributes:
        components: Ordered list of kernel component names (e.g.
            ``["Linear", "StdPeriodic", "White"]``).
        expression: Human-readable additive expression, e.g.
            ``"Linear + StdPeriodic + White"``.
        rationale: One reason string per decision the recommender made.
        confidence: Overall confidence in the suggestion, in ``[0, 1]``.
        characteristics: The :class:`DataCharacteristics` the advice is based on.
        alternatives: Other reasonable expressions worth trying.
    """

    components: List[str]
    expression: str
    rationale: List[str]
    confidence: float
    characteristics: DataCharacteristics
    alternatives: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Render a compact multi-line summary of the recommendation."""
        lines = [
            f"Recommended kernel: {self.expression}",
            f"Confidence: {self.confidence:.0%}",
            "Rationale:",
        ]
        lines += [f"  - {r}" for r in self.rationale]
        if self.alternatives:
            lines.append("Alternatives: " + "; ".join(self.alternatives))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Return the recommendation as a plain dictionary."""
        return {
            "components": self.components,
            "expression": self.expression,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
            "characteristics": self.characteristics.to_dict(),
        }


def _validate_xy(X: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Validate inputs and return sorted 1-D ``(x, y)`` for the first dimension.

    Args:
        X: Inputs, shape ``(n,)``, ``(n, 1)``, or ``(n, d)`` (first column used).
        y: Targets, shape ``(n,)`` or ``(n, 1)``.

    Returns:
        Tuple ``(x, y)`` of finite 1-D arrays sorted by ascending ``x``.

    Raises:
        RecommendationError: If shapes disagree, fewer than 4 points, or values
            are non-finite.
    """
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float).ravel()
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    if Xa.shape[0] != ya.shape[0]:
        raise RecommendationError(
            f"X has {Xa.shape[0]} rows but y has {ya.shape[0]}"
        )
    if Xa.shape[0] < 4:
        raise RecommendationError("need at least 4 points to analyse data")
    if not (np.all(np.isfinite(Xa)) and np.all(np.isfinite(ya))):
        raise RecommendationError("X and y must be finite")
    x = Xa[:, 0]
    order = np.argsort(x)
    return x[order], ya[order]


def analyze_data_characteristics(X: Any, y: Any) -> DataCharacteristics:
    """Estimate trend, periodicity, smoothness, and noise from raw data.

    The estimates are deliberately lightweight heuristics (linear regression,
    a real FFT, and autocorrelation of first differences) chosen to be robust on
    small datasets rather than exact. They feed :func:`suggest_kernel`.

    Args:
        X: Inputs, shape ``(n,)``, ``(n, 1)``, or ``(n, d)`` (first column used).
        y: Targets, shape ``(n,)`` or ``(n, 1)``.

    Returns:
        A populated :class:`DataCharacteristics`.

    Raises:
        RecommendationError: If the inputs are too small or malformed.
    """
    x, yv = _validate_xy(X, y)
    n = x.size
    x_rng = (float(x.min()), float(x.max()))

    # --- Trend: linear fit R^2 and slope
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    slope = float(coef[0])
    y_fit = A @ coef
    ss_res = float(np.sum((yv - y_fit) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2)) or 1e-12
    trend_strength = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))

    # --- Periodicity: FFT of detrended, evenly-resampled signal
    resid = yv - y_fit
    periodicity_strength = 0.0
    dominant_period: Optional[float] = None
    if n >= 8:
        xi = np.linspace(x_rng[0], x_rng[1], n)
        ri = np.interp(xi, x, resid)
        ri = ri - ri.mean()
        spectrum = np.abs(np.fft.rfft(ri))
        if spectrum.size > 2:
            freqs = np.fft.rfftfreq(n, d=(xi[1] - xi[0]))
            peak = int(np.argmax(spectrum[1:]) + 1)  # skip DC
            total = float(np.sum(spectrum[1:])) or 1e-12
            periodicity_strength = float(np.clip(spectrum[peak] / total, 0.0, 1.0))
            if freqs[peak] > 0:
                dominant_period = float(1.0 / freqs[peak])

    # --- Smoothness: lag-1 autocorrelation of first differences
    diffs = np.diff(yv)
    if diffs.size >= 2 and np.std(diffs) > 0:
        ac1 = float(np.corrcoef(diffs[:-1], diffs[1:])[0, 1])
        smoothness = float(np.clip((ac1 + 1.0) / 2.0, 0.0, 1.0))
    else:
        smoothness = 1.0

    # --- Noise: variance of second differences vs signal variance
    d2 = np.diff(yv, n=2)
    noise_var = float(np.var(d2) / 6.0) if d2.size else 0.0
    sig_var = float(np.var(yv)) or 1e-12
    noise_ratio = float(np.clip(noise_var / sig_var, 0.0, 1.0))

    monotonic = bool(np.all(diffs >= -1e-9) or np.all(diffs <= 1e-9))

    return DataCharacteristics(
        n_points=n,
        input_range=x_rng,
        trend_strength=trend_strength,
        trend_slope=slope,
        periodicity_strength=periodicity_strength,
        dominant_period=dominant_period,
        smoothness=smoothness,
        noise_ratio=noise_ratio,
        monotonic=monotonic,
    )


def suggest_kernel(
    X: Any,
    y: Any,
    *,
    trend_threshold: float = 0.6,
    period_threshold: float = 0.25,
    noise_threshold: float = 0.05,
) -> KernelRecommendation:
    """Suggest an additive kernel structure for the given data.

    The rules, in order: a strong linear trend adds a ``Linear`` term; a clear
    dominant cycle adds ``StdPeriodic``; the base stationary term is ``RBF`` for
    smooth signals or ``Matern32`` for rougher ones; appreciable noise adds a
    ``White`` term. Confidence reflects how decisive those signals were.

    Args:
        X: Inputs, shape ``(n,)``, ``(n, 1)``, or ``(n, d)`` (first column used).
        y: Targets, shape ``(n,)`` or ``(n, 1)``.
        trend_threshold: Minimum ``trend_strength`` (R²) to add a ``Linear`` term.
        period_threshold: Minimum ``periodicity_strength`` to add ``StdPeriodic``.
        noise_threshold: Minimum ``noise_ratio`` to add a ``White`` term.

    Returns:
        A :class:`KernelRecommendation`.

    Raises:
        RecommendationError: If the data cannot be analysed.

    Example:
        >>> rec = gpclarity.suggest_kernel(X, y)
        >>> print(rec.expression)
        'RBF + White'
    """
    c = analyze_data_characteristics(X, y)
    components: List[str] = []
    rationale: List[str] = []
    votes: List[float] = []

    if c.trend_strength >= trend_threshold:
        components.append("Linear")
        rationale.append(
            f"Strong linear trend (R²={c.trend_strength:.2f}, "
            f"slope={c.trend_slope:+.3g}) → add Linear."
        )
        votes.append(c.trend_strength)

    if c.periodicity_strength >= period_threshold and c.dominant_period:
        components.append("StdPeriodic")
        rationale.append(
            f"Dominant cycle detected (strength={c.periodicity_strength:.2f}, "
            f"period≈{c.dominant_period:.3g}) → add StdPeriodic."
        )
        votes.append(c.periodicity_strength)

    # base stationary component
    if c.smoothness >= 0.5:
        components.append("RBF")
        rationale.append(
            f"Signal is smooth (smoothness={c.smoothness:.2f}) → RBF base kernel."
        )
    else:
        components.append("Matern32")
        rationale.append(
            f"Signal is rough (smoothness={c.smoothness:.2f}) → Matern32 base kernel."
        )
    votes.append(abs(c.smoothness - 0.5) * 2.0)

    if c.noise_ratio >= noise_threshold:
        components.append("White")
        rationale.append(
            f"Appreciable observation noise (ratio={c.noise_ratio:.2f}) → add White."
        )
        votes.append(min(1.0, c.noise_ratio * 4.0))

    expression = " + ".join(components)
    confidence = float(np.clip(np.mean(votes) if votes else 0.3, 0.1, 0.95))

    alternatives: List[str] = []
    if "RBF" in components:
        alternatives.append(expression.replace("RBF", "Matern52"))
    if "StdPeriodic" not in components and c.periodicity_strength > 0.15:
        alternatives.append(expression + " + StdPeriodic")
    if "Linear" not in components and c.trend_strength > 0.4:
        alternatives.append("Linear + " + expression)

    return KernelRecommendation(
        components=components,
        expression=expression,
        rationale=rationale,
        confidence=confidence,
        characteristics=c,
        alternatives=alternatives,
    )


_KERNEL_BUILDERS = {
    "RBF": lambda gpy, d: gpy.kern.RBF(input_dim=d),
    "Matern32": lambda gpy, d: gpy.kern.Matern32(input_dim=d),
    "Matern52": lambda gpy, d: gpy.kern.Matern52(input_dim=d),
    "Linear": lambda gpy, d: gpy.kern.Linear(input_dim=d),
    "StdPeriodic": lambda gpy, d: gpy.kern.StdPeriodic(input_dim=d),
    "White": lambda gpy, d: gpy.kern.White(input_dim=d),
    "Bias": lambda gpy, d: gpy.kern.Bias(input_dim=d),
}


def build_kernel(recommendation: KernelRecommendation, input_dim: int = 1) -> Any:
    """Construct a concrete ``GPy`` kernel from a recommendation.

    Sums the recommended components into a single ``GPy.kern`` object ready to
    pass to ``GPy.models.GPRegression``. Requires GPy.

    Args:
        recommendation: A :class:`KernelRecommendation` (or any object exposing a
            ``components`` list of known names).
        input_dim: Input dimensionality for the constructed kernels.

    Returns:
        A ``GPy.kern.Kern`` — the additive combination of the components.

    Raises:
        RecommendationError: If GPy is unavailable, ``input_dim < 1``, no
            components are present, or a component name is unknown.

    Example:
        >>> rec = gpclarity.suggest_kernel(X, y)
        >>> kern = gpclarity.build_kernel(rec, input_dim=X.shape[1])
        >>> model = GPy.models.GPRegression(X, y[:, None], kern)
    """
    try:
        import GPy
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RecommendationError(
            "build_kernel requires GPy. Install: pip install gpclarity[full]"
        ) from e

    if input_dim < 1:
        raise RecommendationError("input_dim must be >= 1")
    names = list(getattr(recommendation, "components", []))
    if not names:
        raise RecommendationError("recommendation has no components")

    kern = None
    for name in names:
        builder = _KERNEL_BUILDERS.get(name)
        if builder is None:
            raise RecommendationError(f"unknown kernel component: {name!r}")
        part = builder(GPy, input_dim)
        kern = part if kern is None else kern + part
    return kern
