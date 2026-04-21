"""Hyperparameter tuning with Optuna for XGBoost, LightGBM, and CatBoost."""

from importlib import import_module
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from tabular_ml.config import load_config, resolve_training_backend
from tabular_ml.models.trainer import compute_class_weight

# Suppress Optuna info logging (only show warnings+)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptionalModelDependencyError(ImportError):
    """Raised when an optional native ML dependency cannot be imported."""


_MODEL_IMPORTS = {
    "xgboost": ("xgboost", "XGBClassifier"),
    "lightgbm": ("lightgbm", "LGBMClassifier"),
    "catboost": ("catboost", "CatBoostClassifier"),
}


def _import_model_class(model_type: str):
    """Import a model class only when it is actually required."""
    if model_type not in _MODEL_IMPORTS:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {list(_MODEL_IMPORTS)}"
        )

    module_name, class_name = _MODEL_IMPORTS[model_type]
    try:
        module = import_module(module_name)
    except Exception as exc:  # pragma: no cover - exercised in integration envs
        hint = ""
        if module_name in {"xgboost", "lightgbm"}:
            hint = (
                " On macOS this often means OpenMP is missing; install `libomp` "
                "and retry."
            )
        raise OptionalModelDependencyError(
            f"Could not import optional dependency {module_name!r}.{hint}"
        ) from exc

    return getattr(module, class_name)


def _xgboost_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    fixed_params: dict,
    class_weight: float,
) -> float:
    """Optuna objective for XGBoost hyperparameter search."""
    XGBClassifier = _import_model_class("xgboost")
    params = {
        **fixed_params,
        "scale_pos_weight": class_weight,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }

    model = XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_pred_proba)


def _lightgbm_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    fixed_params: dict,
    class_weight: float,
) -> float:
    """Optuna objective for LightGBM hyperparameter search."""
    LGBMClassifier = _import_model_class("lightgbm")
    params = {
        **fixed_params,
        "scale_pos_weight": class_weight,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 1.0, log=True),
    }

    model = LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
    )
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_pred_proba)


def _catboost_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    fixed_params: dict,
    class_weight: float,
) -> float:
    """Optuna objective for CatBoost hyperparameter search."""
    CatBoostClassifier = _import_model_class("catboost")
    params = {
        **fixed_params,
        "auto_class_weights": "Balanced",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
    }

    model = CatBoostClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
    )
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_pred_proba)


_OBJECTIVES = {
    "xgboost": _xgboost_objective,
    "lightgbm": _lightgbm_objective,
    "catboost": _catboost_objective,
}


def tune_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
) -> dict[str, Any]:
    """Run Optuna hyperparameter search for a given model type.

    Args:
        model_type: One of "xgboost", "lightgbm", "catboost".
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        config: Full configuration dict.

    Returns:
        Dict with best_params (merged fixed + tuned), best_value (PR-AUC), and study.
    """
    training_cfg = config["training"]
    optuna_cfg = training_cfg["optuna"]
    model_cfg = training_cfg["models"][model_type]
    fixed_params = dict(model_cfg["fixed_params"])
    backend = resolve_training_backend(model_type, config)
    fixed_params.update(backend.params)

    class_weight = compute_class_weight(y_train)

    objective_fn = _OBJECTIVES[model_type]

    study = optuna.create_study(
        direction=optuna_cfg["direction"],
        study_name=f"{model_type}-tuning",
    )

    study.optimize(
        lambda trial: objective_fn(
            trial,
            X_train,
            y_train,
            X_val,
            y_val,
            fixed_params,
            class_weight,
        ),
        n_trials=optuna_cfg["n_trials"],
        timeout=optuna_cfg["timeout"],
        show_progress_bar=True,
    )

    # Merge fixed params with best tuned params
    best_params = {**fixed_params, **study.best_params}

    # Add class weight back
    if model_type in ("xgboost", "lightgbm"):
        best_params["scale_pos_weight"] = class_weight
    elif model_type == "catboost":
        best_params["auto_class_weights"] = "Balanced"

    print(f"\n  {model_type.upper()} — Best PR-AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    print(
        "  Hardware backend: "
        f"requested={backend.requested_preference}, "
        f"resolved={backend.resolved_device} ({backend.reason})"
    )

    return {
        "best_params": best_params,
        "best_value": study.best_value,
        "backend": backend.as_dict(),
        "study": study,
    }


def build_model(
    model_type: str,
    params: dict,
    config: dict | None = None,
    backend: dict[str, Any] | None = None,
) -> Any:
    """Instantiate a model with the given parameters.

    Args:
        model_type: One of "xgboost", "lightgbm", "catboost".
        params: Full parameter dict (fixed + tuned).
        config: Optional project config for hardware resolution.
        backend: Optional pre-resolved backend info from ``tune_model``.

    Returns:
        Instantiated model (not yet fitted).
    """
    cfg = config or load_config()
    resolution = backend or resolve_training_backend(model_type, cfg).as_dict()
    init_params = {**resolution["params"], **params}

    model_class = _import_model_class(model_type)
    return model_class(**init_params)
