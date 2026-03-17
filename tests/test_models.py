"""Tests for model evaluation, training, and tuning utilities."""

import numpy as np
import pandas as pd
import pytest

from tabular_ml.models.evaluation import (
    compute_metrics,
    find_optimal_threshold,
    format_results_table,
)
from tabular_ml.models.trainer import compute_class_weight
from tabular_ml.models.tuning import build_model


@pytest.fixture
def binary_predictions():
    """Create synthetic binary classification predictions."""
    rng = np.random.RandomState(42)
    n = 500
    y_true = np.concatenate([np.zeros(490), np.ones(10)])
    # Make probabilities somewhat aligned with truth
    y_proba = rng.uniform(0, 0.3, n)
    y_proba[y_true == 1] = rng.uniform(0.5, 1.0, 10)
    return y_true, y_proba


@pytest.fixture
def sample_train_data():
    """Small synthetic dataset for model instantiation tests."""
    rng = np.random.RandomState(42)
    n = 100
    X = pd.DataFrame({f"f{i}": rng.randn(n) for i in range(5)})
    y = pd.Series(np.concatenate([np.zeros(95), np.ones(5)]), name="Class")
    return X, y


class TestComputeMetrics:
    def test_returns_all_expected_keys(self, binary_predictions):
        y_true, y_proba = binary_predictions
        metrics = compute_metrics(y_true, y_proba)
        expected_keys = {"pr_auc", "roc_auc", "f1", "precision", "recall", "threshold"}
        assert set(metrics.keys()) == expected_keys

    def test_metrics_in_valid_range(self, binary_predictions):
        y_true, y_proba = binary_predictions
        metrics = compute_metrics(y_true, y_proba)
        for key in ["pr_auc", "roc_auc", "f1", "precision", "recall"]:
            assert 0 <= metrics[key] <= 1, f"{key} out of range: {metrics[key]}"

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_proba = np.array([0.0, 0.1, 0.2, 0.9, 1.0])
        metrics = compute_metrics(y_true, y_proba, threshold=0.5)
        assert metrics["pr_auc"] > 0.95
        assert metrics["roc_auc"] > 0.95
        assert metrics["f1"] == 1.0

    def test_custom_threshold(self, binary_predictions):
        y_true, y_proba = binary_predictions
        metrics_high = compute_metrics(y_true, y_proba, threshold=0.9)
        metrics_low = compute_metrics(y_true, y_proba, threshold=0.1)
        # Higher threshold → lower recall, higher precision (generally)
        assert metrics_high["recall"] <= metrics_low["recall"]


class TestFindOptimalThreshold:
    def test_returns_float(self, binary_predictions):
        y_true, y_proba = binary_predictions
        threshold = find_optimal_threshold(y_true, y_proba)
        assert isinstance(threshold, float)

    def test_threshold_in_valid_range(self, binary_predictions):
        y_true, y_proba = binary_predictions
        threshold = find_optimal_threshold(y_true, y_proba)
        assert 0 < threshold < 1

    def test_optimal_threshold_gives_good_f1(self, binary_predictions):
        y_true, y_proba = binary_predictions
        threshold = find_optimal_threshold(y_true, y_proba)
        metrics = compute_metrics(y_true, y_proba, threshold=threshold)
        # Optimal threshold should yield better F1 than default 0.5
        metrics_default = compute_metrics(y_true, y_proba, threshold=0.5)
        assert metrics["f1"] >= metrics_default["f1"]


class TestComputeClassWeight:
    def test_basic_weight(self):
        y = pd.Series([0, 0, 0, 0, 1])
        weight = compute_class_weight(y)
        assert weight == 4.0

    def test_balanced_weight(self):
        y = pd.Series([0, 0, 1, 1])
        weight = compute_class_weight(y)
        assert weight == 1.0

    def test_imbalanced_weight(self):
        y = pd.Series([0] * 577 + [1])
        weight = compute_class_weight(y)
        assert weight == 577.0

    def test_numpy_input(self):
        y = np.array([0, 0, 0, 1])
        weight = compute_class_weight(y)
        assert weight == 3.0


class TestBuildModel:
    def test_xgboost_instantiation(self):
        model = build_model("xgboost", {"n_estimators": 10, "random_state": 42})
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")

    def test_lightgbm_instantiation(self):
        model = build_model(
            "lightgbm", {"n_estimators": 10, "random_state": 42, "verbosity": -1}
        )
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")

    def test_catboost_instantiation(self):
        model = build_model(
            "catboost", {"iterations": 10, "verbose": 0, "random_seed": 42}
        )
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            build_model("unknown_model", {})

    def test_xgboost_can_fit(self, sample_train_data):
        X, y = sample_train_data
        model = build_model(
            "xgboost",
            {
                "n_estimators": 10,
                "random_state": 42,
                "eval_metric": "aucpr",
                "scale_pos_weight": 19.0,
            },
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert (proba >= 0).all() and (proba <= 1).all()


class TestFormatResultsTable:
    def test_table_format(self):
        results = [
            {
                "model_name": "A",
                "pr_auc": 0.9,
                "roc_auc": 0.95,
                "f1": 0.85,
                "precision": 0.8,
                "recall": 0.9,
            },
            {
                "model_name": "B",
                "pr_auc": 0.8,
                "roc_auc": 0.9,
                "f1": 0.75,
                "precision": 0.7,
                "recall": 0.8,
            },
        ]
        table = format_results_table(results)
        assert "| Model |" in table
        assert "| A |" in table
        assert "| B |" in table
        # Should be sorted by PR-AUC descending — A first
        lines = table.strip().split("\n")
        assert "A" in lines[2]
        assert "B" in lines[3]
