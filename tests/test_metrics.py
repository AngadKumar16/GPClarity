"""
Tests for predictive metrics and calibration diagnostics.

These are pure-NumPy and do not require GPy, except ``cross_validate`` which
retrains models per fold.
"""

import numpy as np
import pytest

import gpclarity
from gpclarity.exceptions import MetricError
from gpclarity import metrics


@pytest.fixture
def calibrated():
    """A well-calibrated Gaussian prediction set (mean == truth in expectation)."""
    rng = np.random.default_rng(0)
    n = 2000
    mean = np.zeros(n)
    var = np.ones(n)
    y = mean + rng.standard_normal(n) * np.sqrt(var)
    return y, mean, var


class TestScoringRules:
    def test_nlpd_matches_closed_form(self):
        # single point, y=mean -> nlpd = 0.5*log(2*pi*var)
        val = metrics.nlpd_gaussian([0.0], [0.0], [1.0])
        assert val == pytest.approx(0.5 * np.log(2 * np.pi), rel=1e-6)

    def test_nlpd_penalises_bad_mean(self):
        good = metrics.nlpd_gaussian([0.0], [0.0], [1.0])
        bad = metrics.nlpd_gaussian([3.0], [0.0], [1.0])
        assert bad > good

    def test_crps_nonnegative_and_zero_variance_limit(self):
        val = metrics.crps_gaussian([0.0, 1.0], [0.0, 1.0], [0.25, 0.25])
        assert val >= 0

    def test_crps_perfect_small_variance(self):
        # as variance -> 0 with y == mean, CRPS -> 0
        val = metrics.crps_gaussian([0.0], [0.0], [1e-6])
        assert val < 1e-2

    def test_interval_score_penalises_misses(self):
        inside = metrics.interval_score([0.0], [0.0], [1.0], alpha=0.05)
        outside = metrics.interval_score([10.0], [0.0], [1.0], alpha=0.05)
        assert outside > inside


class TestCalibration:
    def test_coverage_near_nominal_when_calibrated(self, calibrated):
        y, mean, var = calibrated
        cov = metrics.coverage_probability(y, mean, var, confidence_level=1.96)
        assert cov == pytest.approx(0.95, abs=0.03)

    def test_nominal_coverage(self):
        assert metrics.nominal_coverage(1.96) == pytest.approx(0.95, abs=0.001)

    def test_pit_uniform_when_calibrated(self, calibrated):
        y, mean, var = calibrated
        pit = metrics.pit_values(y, mean, var)
        assert pit.min() >= 0.0 and pit.max() <= 1.0
        # mean of uniform ~ 0.5
        assert pit.mean() == pytest.approx(0.5, abs=0.03)

    def test_calibration_error_small_when_calibrated(self, calibrated):
        y, mean, var = calibrated
        assert metrics.calibration_error(y, mean, var) < 0.05

    def test_overconfident_has_low_coverage(self):
        rng = np.random.default_rng(1)
        n = 1000
        y = rng.standard_normal(n)  # true std 1
        mean = np.zeros(n)
        var = np.full(n, 0.01)  # claimed std 0.1 -> overconfident
        assert metrics.coverage_probability(y, mean, var, 1.96) < 0.5

    def test_calibration_curve_shape(self, calibrated):
        y, mean, var = calibrated
        curve = metrics.calibration_curve(y, mean, var, n_bins=10)
        assert curve["expected"].shape == (10,)
        assert curve["observed"].shape == (10,)


class TestSharpnessAndAll:
    def test_sharpness(self):
        assert metrics.sharpness([4.0, 4.0]) == pytest.approx(2.0)

    def test_compute_all_metrics_keys(self, calibrated):
        y, mean, var = calibrated
        d = gpclarity.compute_all_metrics(y, mean, var)
        for k in ("nlpd", "crps", "rmse", "coverage", "calibration_error",
                  "sharpness", "interval_score", "nominal_coverage"):
            assert k in d


class TestValidation:
    def test_shape_mismatch(self):
        with pytest.raises(MetricError):
            metrics.nlpd_gaussian([0, 1], [0], [1])

    def test_nonpositive_variance(self):
        with pytest.raises(MetricError):
            metrics.nlpd_gaussian([0.0], [0.0], [0.0])

    def test_nonfinite(self):
        with pytest.raises(MetricError):
            metrics.nlpd_gaussian([np.nan], [0.0], [1.0])

    def test_bad_alpha(self):
        with pytest.raises(MetricError):
            metrics.interval_score([0.0], [0.0], [1.0], alpha=1.5)


class TestCrossValidation:
    def test_cross_validate(self):
        GPy = pytest.importorskip("GPy")
        rng = np.random.default_rng(3)
        X = np.linspace(0, 10, 40).reshape(-1, 1)
        y = np.sin(X).flatten() + 0.1 * rng.standard_normal(40)

        def factory(Xtr, ytr):
            m = GPy.models.GPRegression(Xtr, ytr, GPy.kern.RBF(1))
            m.optimize()
            return m

        cv = gpclarity.cross_validate(factory, X, y, n_folds=4, random_state=0)
        assert cv.n_folds == 4
        assert len(cv.per_fold) == 4
        assert np.isfinite(cv.nlpd)
        assert 0.0 <= cv.coverage <= 1.0
        assert "rmse" in cv.to_dict()

    def test_cross_validate_bad_folds(self):
        X = np.linspace(0, 1, 5).reshape(-1, 1)
        y = np.zeros(5)
        with pytest.raises(MetricError):
            gpclarity.cross_validate(lambda a, b: None, X, y, n_folds=10)
