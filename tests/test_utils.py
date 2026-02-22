"""
Tests for utility functions.
"""

import GPy
import numpy as np
import pytest

import gpclarity


class TestUtils:
    @pytest.fixture(autouse=True)
    def setup(self, simple_gp):
        self.model = simple_gp

    def test_get_lengthscale(self):
        """Test lengthscale extraction."""
        ls = gpclarity.get_lengthscale(self.model)
        assert isinstance(ls, (float, np.ndarray))
        assert np.all(ls > 0)

    def test_get_noise_variance(self):
        """Test noise variance extraction."""
        noise = gpclarity.get_noise_variance(self.model)
        assert isinstance(noise, float)
        assert noise >= 0

    def test_extract_kernel_params_flat(self):
        """Test flat parameter extraction."""
        params = gpclarity.extract_kernel_params_flat(self.model)
        assert isinstance(params, dict)
        assert len(params) > 0

    def test_check_model_health(self):
        """Test model health checking."""
        # Healthy model
        health = gpclarity.check_model_health(self.model)
        assert isinstance(health, dict)
        assert "healthy" in health
        assert health["healthy"] is True
        assert len(health["issues"]) == 0
        assert "log_likelihood" in health
        
        # Model without predict
        bad_model = object()
        health = gpclarity.check_model_health(bad_model)
        assert health["healthy"] is False
        assert "Model missing predict() method" in health["issues"]

    def test_check_model_health_invalid(self):
        """Test health check on problematic model."""
        # Create model with NaN parameters
        X = np.random.rand(10, 1)
        y = np.random.rand(10)

        kernel = GPy.kern.RBF(1)
        model = GPy.models.GPRegression(X, y[:, None], kernel)
        # Force NaN in parameters
        model.kern.lengthscale = np.nan

        health = gpclarity.check_model_health(model)
        assert health["healthy"] is False