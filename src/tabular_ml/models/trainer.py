"""Model training with MLflow experiment tracking."""

from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd

from tabular_ml.models.evaluation import (
    compute_metrics,
    find_optimal_threshold,
    plot_confusion_matrix,
    plot_precision_recall_curve,
)


def setup_mlflow(config: dict) -> str:
    """Configure MLflow tracking and return the experiment ID.

    Args:
        config: Full configuration dict.

    Returns:
        MLflow experiment ID.
    """
    mlflow_cfg = config["training"]["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    experiment = mlflow.set_experiment(mlflow_cfg["experiment_name"])
    return experiment.experiment_id


def compute_class_weight(y_train: pd.Series | np.ndarray) -> float:
    """Compute scale_pos_weight for imbalanced binary classification.

    Returns the ratio of negative to positive samples.

    Args:
        y_train: Training target labels.

    Returns:
        Weight ratio (n_negative / n_positive).
    """
    y = np.asarray(y_train)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float(n_neg / n_pos)


def train_and_evaluate(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
    config: dict,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Train a model, evaluate on validation set, and log to MLflow.

    Handles the full workflow:
    1. Train the model with early stopping on validation set
    2. Predict probabilities
    3. Find optimal threshold
    4. Compute all metrics
    5. Log everything to MLflow

    Args:
        model: Instantiated sklearn-compatible model (XGBoost/LightGBM/CatBoost).
        model_name: Human-readable name (e.g., "XGBoost").
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        params: Hyperparameters used (for logging).
        config: Full configuration dict.
        log_to_mlflow: Whether to log to MLflow.

    Returns:
        Dict with model_name, metrics, trained model, and optimal threshold.
    """
    artifact_dir = Path(config["pipeline"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Fit with early stopping using eval set
    fit_params = _get_fit_params(model_name, X_val, y_val)
    model.fit(X_train, y_train, **fit_params)

    # Predict
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Find optimal threshold and compute metrics
    optimal_threshold = find_optimal_threshold(y_val.values, y_pred_proba)
    metrics = compute_metrics(y_val.values, y_pred_proba, threshold=optimal_threshold)
    metrics_default = compute_metrics(y_val.values, y_pred_proba, threshold=0.5)

    print(f"\n{'=' * 60}")
    print(f"  {model_name} Results (Validation Set)")
    print(f"{'=' * 60}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f} (threshold={optimal_threshold:.3f})")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"{'=' * 60}")

    if log_to_mlflow:
        _log_to_mlflow(
            model=model,
            model_name=model_name,
            params=params,
            metrics=metrics,
            metrics_default=metrics_default,
            optimal_threshold=optimal_threshold,
            y_val=y_val,
            y_pred_proba=y_pred_proba,
            artifact_dir=artifact_dir,
        )

    # Save model artifact
    model_path = artifact_dir / f"{model_name.lower().replace(' ', '_')}_model.joblib"
    joblib.dump(model, model_path)
    print(f"  Model saved to {model_path}")

    return {
        "model_name": model_name,
        "model": model,
        "optimal_threshold": optimal_threshold,
        **metrics,
    }


def _get_fit_params(model_name: str, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Build model-specific fit parameters for early stopping.

    Args:
        model_name: One of "XGBoost", "LightGBM", "CatBoost".
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Dict of keyword arguments for model.fit().
    """
    name = model_name.lower()
    if "xgboost" in name or "xgb" in name:
        return {"eval_set": [(X_val, y_val)], "verbose": False}
    elif "lightgbm" in name or "lgbm" in name:
        return {"eval_set": [(X_val, y_val)]}
    elif "catboost" in name or "cat" in name:
        return {"eval_set": (X_val, y_val)}
    else:
        return {}


def _log_to_mlflow(
    model: Any,
    model_name: str,
    params: dict,
    metrics: dict,
    metrics_default: dict,
    optimal_threshold: float,
    y_val: pd.Series | np.ndarray,
    y_pred_proba: np.ndarray,
    artifact_dir: Path,
) -> None:
    """Log everything to an MLflow run.

    Args:
        model: Trained model.
        model_name: Human-readable model name.
        params: Hyperparameters.
        metrics: Metrics at optimal threshold.
        metrics_default: Metrics at default 0.5 threshold.
        optimal_threshold: The optimal threshold found.
        y_val: Validation true labels.
        y_pred_proba: Predicted probabilities.
        artifact_dir: Directory for saving plot artifacts.
    """
    with mlflow.start_run(run_name=model_name):
        # Log params
        mlflow.log_params({f"param_{k}": v for k, v in params.items()})

        # Log metrics at optimal threshold
        mlflow.log_metrics(
            {
                "val_pr_auc": metrics["pr_auc"],
                "val_roc_auc": metrics["roc_auc"],
                "val_f1": metrics["f1"],
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "optimal_threshold": optimal_threshold,
            }
        )

        # Also log metrics at default 0.5 threshold for comparison
        mlflow.log_metrics(
            {
                "val_f1_t05": metrics_default["f1"],
                "val_precision_t05": metrics_default["precision"],
                "val_recall_t05": metrics_default["recall"],
            }
        )

        # Generate and log plots
        y_true = np.asarray(y_val)
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)

        pr_path = artifact_dir / f"{model_name.lower().replace(' ', '_')}_pr_curve.png"
        plot_precision_recall_curve(y_true, y_pred_proba, model_name, save_path=pr_path)
        mlflow.log_artifact(str(pr_path))

        cm_path = (
            artifact_dir
            / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        )
        plot_confusion_matrix(y_true, y_pred, model_name, save_path=cm_path)
        mlflow.log_artifact(str(cm_path))

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")
