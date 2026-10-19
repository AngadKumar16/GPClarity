"""
Tests for model comparison and selection.

These require GPy for real trained models, but the ranking logic is also checked
against lightweight fakes.
"""

import numpy as np
import pytest

import gpclarity
from gpclarity.exceptions import ComparisonError
from gpclarity import model_comparison as mc


class _FakeModel:
    """Minimal duck-typed stand-in for a trained GP model."""

    def __init__(self, ll, n_params, n_data=50, pred=(0.0, 1.0)):
        self._ll = ll
        self.optimizer_array = np.zeros(n_params)
        self.X = np.zeros((n_data, 1))
        self._pred = pred

    def log_likelihood(self):
        return self._ll

    def predict(self, X):
        n = np.asarray(X).shape[0]
        return np.full((n, 1), self._pred[0]), np.full((n, 1), self._pred[1])


class TestScoring:
    def test_aic_bic_formula(self):
        m = _FakeModel(ll=-10.0, n_params=3, n_data=100)
        s = mc.score_model("m", m)
        assert s.aic == pytest.approx(2 * 3 - 2 * (-10.0))
        assert s.bic == pytest.approx(3 * np.log(100) - 2 * (-10.0))
        assert s.n_params == 3

    def test_param_count_fallback(self):
        m = _FakeModel(ll=-1.0, n_params=5)
        assert mc.score_model("m", m).n_params == 5


class TestCompare:
    def test_rank_by_bic(self):
        models = {
            "simple": _FakeModel(ll=-20.0, n_params=1),
            "good": _FakeModel(ll=-5.0, n_params=2),
            "overfit": _FakeModel(ll=-4.5, n_params=20),
        }
        cmp = gpclarity.compare_models(models, criterion="bic")
        assert cmp.best == "good"
        assert cmp.ranking[0] == "good"
        assert set(cmp.ranking) == set(models)

    def test_rank_by_log_likelihood_higher_better(self):
        models = {
            "a": _FakeModel(ll=-5.0, n_params=2),
            "b": _FakeModel(ll=-1.0, n_params=2),
        }
        cmp = gpclarity.compare_models(models, criterion="log_likelihood")
        assert cmp.best == "b"

    def test_select_best_model(self):
        models = {
            "a": _FakeModel(ll=-5.0, n_params=2),
            "b": _FakeModel(ll=-50.0, n_params=2),
        }
        name, model = gpclarity.select_best_model(models, criterion="aic")
        assert name == "a"
        assert model is models["a"]

    def test_test_metrics_criterion(self):
        models = {
            "a": _FakeModel(ll=-5.0, n_params=2, pred=(0.0, 1.0)),
            "b": _FakeModel(ll=-5.0, n_params=2, pred=(5.0, 1.0)),
        }
        X_test = np.zeros((10, 1))
        y_test = np.zeros(10)
        cmp = gpclarity.compare_models(
            models, X_test=X_test, y_test=y_test, criterion="rmse"
        )
        assert cmp.best == "a"  # predicts closer to zero

    def test_table_and_dict(self):
        models = {
            "a": _FakeModel(ll=-5.0, n_params=2),
            "b": _FakeModel(ll=-9.0, n_params=2),
        }
        cmp = gpclarity.compare_models(models, criterion="bic")
        assert "a" in cmp.to_table()
        d = cmp.to_dict()
        assert d["best"] == cmp.best and len(d["scores"]) == 2


class TestValidation:
    def test_one_model(self):
        with pytest.raises(ComparisonError):
            gpclarity.compare_models({"a": _FakeModel(-1.0, 1)})

    def test_bad_criterion(self):
        models = {"a": _FakeModel(-1.0, 1), "b": _FakeModel(-2.0, 1)}
        with pytest.raises(ComparisonError):
            gpclarity.compare_models(models, criterion="nope")

    def test_test_criterion_without_data(self):
        models = {"a": _FakeModel(-1.0, 1), "b": _FakeModel(-2.0, 1)}
        with pytest.raises(ComparisonError):
            gpclarity.compare_models(models, criterion="nlpd")


class TestWithRealModels:
    def test_compare_real_gp(self, simple_gp, composite_gp):
        cmp = gpclarity.compare_models(
            {"rbf": simple_gp, "rbf+white": composite_gp}, criterion="bic"
        )
        assert cmp.best in ("rbf", "rbf+white")
        assert all(np.isfinite(s.log_likelihood) for s in cmp.scores)
