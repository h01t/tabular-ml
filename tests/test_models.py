"""Tests for model evaluation, backend resolution, and optional model builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabular_ml.config import resolve_training_backend
from tabular_ml.models.evaluation import (
    compute_metrics,
    find_optimal_threshold,
    format_results_table,
)
from tabular_ml.models.trainer import compute_class_weight
from tabular_ml.models.tuning import build_model
from tests.conftest import optional_dependency_available


XGBOOST_AVAILABLE = optional_dependency_available("xgboost")
LIGHTGBM_AVAILABLE = optional_dependency_available("lightgbm")
CATBOOST_AVAILABLE = optional_dependency_available("catboost")


@pytest.fixture
def binary_predictions():
    """Create synthetic binary classification predictions."""
    rng = np.random.RandomState(42)
    n = 500
    y_true = np.concatenate([np.zeros(490), np.ones(10)])
    y_proba = rng.uniform(0, 0.3, n)
    y_proba[y_true == 1] = rng.uniform(0.5, 1.0, 10)
    return y_true, y_proba


@pytest.fixture
def sample_train_data():
    """Small synthetic dataset for optional model-instantiation tests."""
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


class TestTrainingBackendResolution:
    @pytest.fixture
    def base_config(self):
        return {
            "training": {
                "hardware": {
                    "preference": "auto",
                    "auto_default_device": "cpu",
                    "model_parameters": {
                        "xgboost": {
                            "cpu": {},
                            "gpu": {"device": "cuda"},
                        },
                        "lightgbm": {
                            "cpu": {"device_type": "cpu"},
                            "gpu": {"device_type": "gpu"},
                        },
                        "catboost": {
                            "cpu": {"task_type": "CPU"},
                            "gpu": {"task_type": "GPU"},
                        },
                    },
                }
            }
        }

    def test_auto_uses_cpu_on_apple_silicon(self, base_config):
        resolution = resolve_training_backend(
            "xgboost",
            base_config,
            system_name="Darwin",
            machine_name="arm64",
        )
        assert resolution.resolved_device == "cpu"
        assert resolution.reason == "auto_cpu_on_apple_silicon"

    def test_explicit_gpu_falls_back_on_apple_silicon(self, base_config):
        base_config["training"]["hardware"]["preference"] = "gpu"
        resolution = resolve_training_backend(
            "lightgbm",
            base_config,
            system_name="Darwin",
            machine_name="arm64",
        )
        assert resolution.resolved_device == "cpu"
        assert resolution.params == {"device_type": "cpu"}

    def test_explicit_gpu_uses_gpu_mapping_when_supported(self, base_config):
        base_config["training"]["hardware"]["preference"] = "gpu"
        resolution = resolve_training_backend(
            "catboost",
            base_config,
            system_name="Linux",
            machine_name="x86_64",
        )
        assert resolution.resolved_device == "gpu"
        assert resolution.params == {"task_type": "GPU"}


class TestBuildModel:
    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            build_model("unknown_model", {})

    @pytest.mark.integration
    @pytest.mark.skipif(
        not XGBOOST_AVAILABLE,
        reason="xgboost unavailable in this environment (likely missing libomp)",
    )
    def test_xgboost_instantiation(self):
        model = build_model("xgboost", {"n_estimators": 10, "random_state": 42})
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not LIGHTGBM_AVAILABLE,
        reason="lightgbm unavailable in this environment (likely missing libomp)",
    )
    def test_lightgbm_instantiation(self):
        model = build_model(
            "lightgbm",
            {"n_estimators": 10, "random_state": 42, "verbosity": -1},
        )
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not CATBOOST_AVAILABLE,
        reason="catboost unavailable in this environment",
    )
    def test_catboost_instantiation(self):
        model = build_model(
            "catboost",
            {"iterations": 10, "verbose": 0, "random_seed": 42},
        )
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not XGBOOST_AVAILABLE,
        reason="xgboost unavailable in this environment (likely missing libomp)",
    )
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
        lines = table.strip().split("\n")
        assert "A" in lines[2]
        assert "B" in lines[3]
