"""Shared configuration and runtime helpers for the tabular-ml project."""

from __future__ import annotations

from dataclasses import dataclass
from os import getenv
from pathlib import Path
from platform import machine, system
from typing import Any

import yaml

from tabular_ml import __version__


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve the active YAML configuration path.

    Priority order:
    1. Explicit function argument
    2. ``TABULAR_ML_CONFIG_PATH`` environment variable
    3. Repository default config
    """
    if config_path is not None:
        return Path(config_path)

    env_path = getenv("TABULAR_ML_CONFIG_PATH")
    if env_path:
        return Path(env_path)

    return DEFAULT_CONFIG_PATH


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the active YAML configuration file."""
    with open(resolve_config_path(config_path)) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class HardwareResolution:
    """Resolved training backend for a specific model family."""

    model_type: str
    requested_preference: str
    resolved_device: str
    reason: str
    params: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the resolution."""
        return {
            "model_type": self.model_type,
            "requested_preference": self.requested_preference,
            "resolved_device": self.resolved_device,
            "reason": self.reason,
            "params": dict(self.params),
        }


def resolve_training_backend(
    model_type: str,
    config: dict[str, Any] | None = None,
    *,
    system_name: str | None = None,
    machine_name: str | None = None,
) -> HardwareResolution:
    """Resolve the training backend for a model family.

    The project supports explicit ``cpu`` / ``gpu`` preferences plus an ``auto``
    mode. ``auto`` currently resolves to CPU for safety and reproducibility, and
    Apple Silicon always resolves to CPU because the active learner stack does
    not provide a native MPS backend.
    """
    cfg = config or load_config()
    hardware_cfg = cfg.get("training", {}).get("hardware", {})

    requested = getenv(
        "TABULAR_ML_DEVICE_PREFERENCE",
        hardware_cfg.get("preference", "auto"),
    ).lower()
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError(
            "Unsupported training.hardware.preference "
            f"{requested!r}. Expected one of: auto, cpu, gpu."
        )

    current_system = system_name or system()
    current_machine = (machine_name or machine()).lower()
    is_apple_silicon = current_system == "Darwin" and current_machine in {
        "arm64",
        "aarch64",
    }

    if requested == "cpu":
        resolved_device = "cpu"
        reason = "explicit_cpu"
    elif requested == "gpu" and is_apple_silicon:
        resolved_device = "cpu"
        reason = "apple_silicon_uses_cpu_for_current_tree_models"
    elif requested == "gpu":
        resolved_device = "gpu"
        reason = "explicit_gpu"
    elif is_apple_silicon:
        resolved_device = "cpu"
        reason = "auto_cpu_on_apple_silicon"
    else:
        resolved_device = hardware_cfg.get("auto_default_device", "cpu")
        reason = "auto_default"

    model_params = (
        hardware_cfg.get("model_parameters", {})
        .get(model_type, {})
        .get(resolved_device, {})
    )

    return HardwareResolution(
        model_type=model_type,
        requested_preference=requested,
        resolved_device=resolved_device,
        reason=reason,
        params=dict(model_params),
    )


def get_serving_settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return serving settings with sensible defaults."""
    cfg = config or load_config()
    serving_cfg = cfg.get("serving", {})
    api_cfg = serving_cfg.get("api", {})
    artifact_cfg = serving_cfg.get("artifacts", {})
    model_cfg = serving_cfg.get("model", {})

    return {
        "api_title": api_cfg.get("title", "Fraud Detection API"),
        "api_description": api_cfg.get(
            "description",
            "Credit card fraud detection inference service.",
        ),
        "api_version": api_cfg.get("version", __version__),
        "pipeline_path": Path(
            artifact_cfg.get("pipeline_path", "artifacts/preprocessing_pipeline.joblib")
        ),
        "model_path": Path(
            artifact_cfg.get("model_path", "artifacts/xgboost_model.joblib")
        ),
        "model_name": model_cfg.get("name", "XGBoost"),
        "model_version": model_cfg.get("version", __version__),
        "threshold": float(model_cfg.get("threshold", 0.893)),
    }
