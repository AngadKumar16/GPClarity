"""
Tests for data-driven kernel recommendation.

Data analysis is pure NumPy; ``build_kernel`` requires GPy.
"""

import numpy as np
import pytest

import gpclarity
from gpclarity.exceptions import RecommendationError
from gpclarity import kernel_recommendation as kr


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 80)
    y = 3.0 * x + 1.0 + 0.05 * rng.standard_normal(80)
    return x, y


@pytest.fixture
def periodic_data():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 20, 200)
    y = np.sin(2 * np.pi * x / 4.0) + 0.02 * rng.standard_normal(200)
    return x, y


@pytest.fixture
def noisy_data():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 10, 120)
    y = np.sin(x) + 0.8 * rng.standard_normal(120)
    return x, y


class TestAnalysis:
    def test_linear_trend_detected(self, linear_data):
        x, y = linear_data
        c = gpclarity.analyze_data_characteristics(x, y)
        assert c.trend_strength > 0.9
        assert c.trend_slope == pytest.approx(3.0, rel=0.1)
        assert c.monotonic

    def test_periodicity_detected(self, periodic_data):
        x, y = periodic_data
        c = gpclarity.analyze_data_characteristics(x, y)
        assert c.periodicity_strength > 0.2
        assert c.dominant_period == pytest.approx(4.0, rel=0.3)

    def test_noise_detected(self, noisy_data):
        x, y = noisy_data
        c = gpclarity.analyze_data_characteristics(x, y)
        assert c.noise_ratio > 0.05

    def test_to_dict(self, linear_data):
        x, y = linear_data
        d = gpclarity.analyze_data_characteristics(x, y).to_dict()
        assert "trend_strength" in d and "noise_ratio" in d


class TestSuggest:
    def test_linear_suggests_linear_component(self, linear_data):
        x, y = linear_data
        rec = gpclarity.suggest_kernel(x, y)
        assert "Linear" in rec.components
        assert 0.0 <= rec.confidence <= 1.0

    def test_periodic_suggests_periodic(self, periodic_data):
        x, y = periodic_data
        rec = gpclarity.suggest_kernel(x, y)
        assert "StdPeriodic" in rec.components

    def test_noisy_suggests_white(self, noisy_data):
        x, y = noisy_data
        rec = gpclarity.suggest_kernel(x, y)
        assert "White" in rec.components

    def test_expression_and_str(self, linear_data):
        x, y = linear_data
        rec = gpclarity.suggest_kernel(x, y)
        assert rec.expression == " + ".join(rec.components)
        assert "Recommended kernel" in str(rec)
        assert "components" in rec.to_dict()

    def test_has_base_kernel(self, noisy_data):
        x, y = noisy_data
        rec = gpclarity.suggest_kernel(x, y)
        assert any(k in rec.components for k in ("RBF", "Matern32"))


class TestValidation:
    def test_too_few_points(self):
        with pytest.raises(RecommendationError):
            gpclarity.analyze_data_characteristics([1, 2], [1, 2])

    def test_shape_mismatch(self):
        with pytest.raises(RecommendationError):
            gpclarity.analyze_data_characteristics(np.arange(5), np.arange(6))


class TestBuildKernel:
    def test_build_kernel(self, linear_data):
        GPy = pytest.importorskip("GPy")
        x, y = linear_data
        rec = gpclarity.suggest_kernel(x, y)
        kern = gpclarity.build_kernel(rec, input_dim=1)
        assert kern is not None
        # Usable in a real model
        model = GPy.models.GPRegression(x.reshape(-1, 1), y.reshape(-1, 1), kern)
        assert model is not None

    def test_build_kernel_unknown_component(self):
        pytest.importorskip("GPy")
        rec = kr.KernelRecommendation(
            components=["NoSuchKernel"],
            expression="NoSuchKernel",
            rationale=[],
            confidence=0.5,
            characteristics=gpclarity.analyze_data_characteristics(
                np.arange(10.0), np.arange(10.0)
            ),
        )
        with pytest.raises(RecommendationError):
            gpclarity.build_kernel(rec)
