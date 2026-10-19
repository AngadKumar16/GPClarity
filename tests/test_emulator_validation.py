"""
Tests for Bastos & O'Hagan emulator validation diagnostics.
"""

import GPy
import numpy as np
import pytest

import gpclarity
from gpclarity.emulator_validation import (
    EmulatorValidation,
    emulator_diagnostics,
    validate_emulator,
)
from gpclarity.exceptions import ValidationError


def _rbf_cov(n, lengthscale=0.3, variance=1.0, noise=1e-2):
    """Squared-exponential covariance on a 1-D grid plus a small nugget."""
    x = np.linspace(0, 1, n)
    d = x[:, None] - x[None, :]
    return variance * np.exp(-0.5 * (d / lengthscale) ** 2) + noise * np.eye(n)


@pytest.fixture
def correlated_draw():
    """One draw from a correlated Gaussian with its true mean and covariance."""
    rng = np.random.default_rng(1)
    n = 40
    V = _rbf_cov(n)
    mean = np.zeros(n)
    y = rng.multivariate_normal(mean, V)
    return y, mean, V


class TestEmulatorDiagnostics:
    def test_returns_result(self, correlated_draw):
        y, mean, V = correlated_draw
        res = emulator_diagnostics(y, mean, V)
        assert isinstance(res, EmulatorValidation)
        assert res.n_points == 40
        assert res.standardized_errors.shape == (40,)
        assert res.pivoted_cholesky_errors.shape == (40,)
        assert sorted(res.pivot_order.tolist()) == list(range(40))

    def test_mahalanobis_matches_direct_formula(self, correlated_draw):
        y, mean, V = correlated_draw
        res = emulator_diagnostics(y, mean, V)
        r = y - mean
        direct = float(r @ np.linalg.solve(V, r))
        assert res.mahalanobis == pytest.approx(direct, rel=1e-8)

    def test_reference_moments(self, correlated_draw):
        y, mean, V = correlated_draw
        res = emulator_diagnostics(y, mean, V)
        assert res.mahalanobis_expected == 40
        assert res.mahalanobis_std == pytest.approx(np.sqrt(80))

    def test_standardized_errors(self, correlated_draw):
        y, mean, V = correlated_draw
        res = emulator_diagnostics(y, mean, V)
        np.testing.assert_allclose(
            res.standardized_errors, (y - mean) / np.sqrt(np.diag(V))
        )

    def test_pivot_order_starts_with_largest_variance(self):
        V = np.diag([0.5, 3.0, 1.0])
        res = emulator_diagnostics([0.1, 0.2, 0.3], [0.0, 0.0, 0.0], V)
        assert res.pivot_order[0] == 1

    def test_calibrated_is_consistent(self, correlated_draw):
        y, mean, V = correlated_draw
        res = emulator_diagnostics(y, mean, V)
        assert res.verdict == "consistent"
        assert res.mahalanobis_pvalue >= 0.05

    def test_shrunk_covariance_is_overconfident(self, correlated_draw):
        y, mean, V = correlated_draw
        res = emulator_diagnostics(y, mean, 0.1 * V)
        assert res.verdict == "overconfident"
        assert res.mahalanobis > res.mahalanobis_expected

    def test_inflated_covariance_is_underconfident(self, correlated_draw):
        y, mean, V = correlated_draw
        res = emulator_diagnostics(y, mean, 10.0 * V)
        assert res.verdict == "underconfident"
        assert res.mahalanobis < res.mahalanobis_expected

    @staticmethod
    def _residuals_with_pc_errors(V, pc_errors):
        """Build residuals whose pivoted Cholesky errors equal ``pc_errors``."""
        from scipy.linalg import lapack

        c, piv, _, _ = lapack.dpstrf(V, lower=1)
        G = np.tril(c)
        resid = np.empty(len(pc_errors))
        resid[piv - 1] = G @ pc_errors
        return resid

    @pytest.mark.parametrize(
        "where, phrase",
        [("late", "late in the pivot order"), ("early", "early in the pivot order")],
    )
    def test_pivot_pattern_interpretation(self, where, phrase):
        V = _rbf_cov(20)
        e = np.full(20, 0.5)
        if where == "late":
            e[-4:] = 4.0
        else:
            e[:4] = 4.0
        resid = self._residuals_with_pc_errors(V, e)
        res = emulator_diagnostics(resid, np.zeros(20), V)
        np.testing.assert_allclose(res.pivoted_cholesky_errors, e, atol=1e-8)
        assert any(phrase in s for s in res.interpretation)

    def test_uniformly_large_errors_read_as_variance(self, correlated_draw):
        y, mean, V = correlated_draw
        res = emulator_diagnostics(y, mean, V / 9.0)
        assert any("spread across" in s for s in res.interpretation)

    def test_to_dict_is_serialisable(self, correlated_draw):
        import json

        y, mean, V = correlated_draw
        json.dumps(emulator_diagnostics(y, mean, V).to_dict())

    def test_column_vectors_accepted(self, correlated_draw):
        y, mean, V = correlated_draw
        res = emulator_diagnostics(y[:, None], mean[:, None], V)
        assert res.n_points == 40

    @pytest.mark.parametrize(
        "y, mean, cov, msg",
        [
            ([1.0], [0.0], [[1.0]], "at least 2"),
            ([1.0, 2.0], [0.0], np.eye(2), "shape mismatch"),
            ([1.0, 2.0], [0.0, 0.0], np.eye(3), "covariance must be"),
            ([1.0, np.nan], [0.0, 0.0], np.eye(2), "non-finite"),
            ([1.0, 2.0], [0.0, 0.0], [[1.0, 0.5], [0.0, 1.0]], "not symmetric"),
            ([1.0, 2.0], [0.0, 0.0], [[0.0, 0.0], [0.0, 1.0]], "non-positive"),
        ],
    )
    def test_invalid_inputs(self, y, mean, cov, msg):
        with pytest.raises(ValidationError, match=msg):
            emulator_diagnostics(y, mean, cov)

    def test_invalid_alpha(self, correlated_draw):
        y, mean, V = correlated_draw
        with pytest.raises(ValidationError, match="alpha"):
            emulator_diagnostics(y, mean, V, alpha=1.5)


class TestValidateEmulator:
    def test_gpy_model_on_held_out_data(self, simple_gp):
        rng = np.random.default_rng(7)
        X_val = np.linspace(0.25, 9.75, 15).reshape(-1, 1)
        y_val = np.sin(X_val).ravel() + 0.1 * rng.standard_normal(15)
        res = validate_emulator(simple_gp, X_val, y_val)
        assert res.n_points == 15
        assert res.verdict in {"consistent", "overconfident", "underconfident"}
        assert res.interpretation

    def test_1d_inputs_reshaped(self, simple_gp):
        X_val = np.linspace(0.25, 9.75, 10)
        y_val = np.sin(X_val)
        assert validate_emulator(simple_gp, X_val, y_val).n_points == 10

    def test_noise_inflates_covariance(self, simple_gp):
        X_val = np.linspace(0.25, 9.75, 10).reshape(-1, 1)
        y_val = np.sin(X_val).ravel()
        with_noise = validate_emulator(simple_gp, X_val, y_val)
        without = validate_emulator(simple_gp, X_val, y_val, include_likelihood=False)
        assert with_noise.mahalanobis < without.mahalanobis

    def test_rejects_object_without_predict(self):
        with pytest.raises(ValidationError, match="predict"):
            validate_emulator(object(), np.zeros((2, 1)), np.zeros(2))

    def test_exported_from_package(self):
        assert gpclarity.validate_emulator is validate_emulator
        assert gpclarity.emulator_diagnostics is emulator_diagnostics
