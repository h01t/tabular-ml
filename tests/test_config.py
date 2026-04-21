"""Tests for shared config helpers."""

from pathlib import Path

from tabular_ml.config import get_serving_settings, load_config, resolve_config_path


def test_resolve_config_path_defaults_to_repo_config():
    path = resolve_config_path()
    assert path == Path("configs/default.yaml")


def test_load_config_includes_serving_and_hardware_sections():
    config = load_config()
    assert "training" in config
    assert "hardware" in config["training"]
    assert "serving" in config


def test_get_serving_settings_uses_config_values():
    settings = get_serving_settings()
    assert settings["api_title"] == "Fraud Detection API"
    assert settings["model_name"] == "XGBoost"
    assert settings["pipeline_path"].name == "preprocessing_pipeline.joblib"
