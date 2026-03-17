"""Tests for stacking and blending ensemble methods."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from tabular_ml.models.ensemble import BlendingEnsemble, StackingEnsemble


@pytest.fixture
def fitted_base_models():
    """Train small base models on synthetic data for ensemble testing."""
    rng = np.random.RandomState(42)
    n = 300
    X = pd.DataFrame({f"f{i}": rng.randn(n) for i in range(5)})
    y = pd.Series(np.concatenate([np.zeros(285), np.ones(15)]), name="Class")

    model_a = XGBClassifier(n_estimators=10, random_state=42, eval_metric="logloss")
    model_b = XGBClassifier(n_estimators=20, random_state=123, eval_metric="logloss")
    model_a.fit(X, y)
    model_b.fit(X, y)

    base_models = [("ModelA", model_a), ("ModelB", model_b)]
    return base_models, X, y


@pytest.fixture
def three_base_models():
    """Train 3 small base models for full ensemble testing."""
    rng = np.random.RandomState(42)
    n = 300
    X = pd.DataFrame({f"f{i}": rng.randn(n) for i in range(5)})
    y = pd.Series(np.concatenate([np.zeros(285), np.ones(15)]), name="Class")

    models = []
    for i, (n_est, seed) in enumerate([(10, 42), (20, 123), (15, 99)]):
        m = XGBClassifier(n_estimators=n_est, random_state=seed, eval_metric="logloss")
        m.fit(X, y)
        models.append((f"Model{i}", m))

    return models, X, y


class TestStackingEnsemble:
    def test_fit_and_predict(self, fitted_base_models):
        X_train = fitted_base_models[1]
        y_train = fitted_base_models[2]
        base_models = fitted_base_models[0]

        ensemble = StackingEnsemble(base_models, n_folds=3)
        ensemble.fit(X_train, y_train)

        proba = ensemble.predict_proba(X_train)
        assert proba.shape == (len(X_train), 2)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_binary(self, fitted_base_models):
        X_train = fitted_base_models[1]
        y_train = fitted_base_models[2]
        base_models = fitted_base_models[0]

        ensemble = StackingEnsemble(base_models, n_folds=3)
        ensemble.fit(X_train, y_train)

        preds = ensemble.predict(X_train)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_base_weights_stored(self, fitted_base_models):
        X_train = fitted_base_models[1]
        y_train = fitted_base_models[2]
        base_models = fitted_base_models[0]

        ensemble = StackingEnsemble(base_models, n_folds=3)
        ensemble.fit(X_train, y_train)

        assert hasattr(ensemble, "base_weights_")
        assert "ModelA" in ensemble.base_weights_
        assert "ModelB" in ensemble.base_weights_

    def test_unfitted_raises(self, fitted_base_models):
        base_models = fitted_base_models[0]
        ensemble = StackingEnsemble(base_models)
        with pytest.raises(RuntimeError, match="not fitted"):
            ensemble.predict_proba(fitted_base_models[1])

    def test_save_and_load(self, fitted_base_models, tmp_path):
        X_train = fitted_base_models[1]
        y_train = fitted_base_models[2]
        base_models = fitted_base_models[0]

        ensemble = StackingEnsemble(base_models, n_folds=3)
        ensemble.fit(X_train, y_train)

        path = tmp_path / "ensemble.joblib"
        ensemble.save(path)

        loaded = StackingEnsemble.load(path)
        original_proba = ensemble.predict_proba(X_train)
        loaded_proba = loaded.predict_proba(X_train)
        np.testing.assert_array_almost_equal(original_proba, loaded_proba)

    def test_three_models(self, three_base_models):
        models, X, y = three_base_models
        ensemble = StackingEnsemble(models, n_folds=3)
        ensemble.fit(X, y)

        proba = ensemble.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert len(ensemble.base_weights_) == 3


class TestBlendingEnsemble:
    def test_fit_and_predict(self, fitted_base_models):
        X_val = fitted_base_models[1]
        y_val = fitted_base_models[2]
        base_models = fitted_base_models[0]

        ensemble = BlendingEnsemble(base_models)
        ensemble.fit(X_val, y_val, n_steps=11)

        proba = ensemble.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_weights_sum_to_one(self, fitted_base_models):
        X_val = fitted_base_models[1]
        y_val = fitted_base_models[2]
        base_models = fitted_base_models[0]

        ensemble = BlendingEnsemble(base_models)
        ensemble.fit(X_val, y_val, n_steps=11)

        assert abs(ensemble.weights_.sum() - 1.0) < 1e-6

    def test_three_model_weights(self, three_base_models):
        models, X, y = three_base_models
        ensemble = BlendingEnsemble(models)
        ensemble.fit(X, y, n_steps=11)

        assert len(ensemble.weights_) == 3
        assert abs(ensemble.weights_.sum() - 1.0) < 1e-6

    def test_predict_binary(self, fitted_base_models):
        X_val = fitted_base_models[1]
        y_val = fitted_base_models[2]
        base_models = fitted_base_models[0]

        ensemble = BlendingEnsemble(base_models)
        ensemble.fit(X_val, y_val, n_steps=11)

        preds = ensemble.predict(X_val)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_unfitted_raises(self, fitted_base_models):
        base_models = fitted_base_models[0]
        ensemble = BlendingEnsemble(base_models)
        with pytest.raises(RuntimeError, match="not fitted"):
            ensemble.predict_proba(fitted_base_models[1])

    def test_save_and_load(self, fitted_base_models, tmp_path):
        X_val = fitted_base_models[1]
        y_val = fitted_base_models[2]
        base_models = fitted_base_models[0]

        ensemble = BlendingEnsemble(base_models)
        ensemble.fit(X_val, y_val, n_steps=11)

        path = tmp_path / "blend.joblib"
        ensemble.save(path)

        loaded = BlendingEnsemble.load(path)
        original_proba = ensemble.predict_proba(X_val)
        loaded_proba = loaded.predict_proba(X_val)
        np.testing.assert_array_almost_equal(original_proba, loaded_proba)
