"""Run ensemble on pre-trained models — stacking and blending comparison."""

import json
from pathlib import Path

import joblib
import mlflow
import numpy as np

from tabular_ml.features.pipeline import fit_and_transform, load_config, save_pipeline
from tabular_ml.models.ensemble import BlendingEnsemble, StackingEnsemble
from tabular_ml.models.evaluation import (
    compute_metrics,
    find_optimal_threshold,
    format_results_table,
    plot_confusion_matrix,
    plot_precision_recall_curve,
)
from tabular_ml.models.trainer import setup_mlflow


def run_ensemble(config_path: str = "configs/default.yaml") -> dict:
    """Build and evaluate stacking + blending ensembles from pre-trained models.

    Loads the three saved base models (XGBoost, LightGBM, CatBoost),
    builds both a stacking and blending ensemble, and compares all
    five approaches.

    Args:
        config_path: Path to YAML config.

    Returns:
        Dict with ensemble results and comparison.
    """
    config = load_config(config_path)
    artifact_dir = Path(config["pipeline"]["artifact_dir"])

    # Load pre-trained base models
    print("=" * 70)
    print("  Loading pre-trained base models")
    print("=" * 70)

    model_files = {
        "XGBoost": artifact_dir / "xgboost_model.joblib",
        "LightGBM": artifact_dir / "lightgbm_model.joblib",
        "CatBoost": artifact_dir / "catboost_model.joblib",
    }

    base_models = []
    for name, path in model_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Base model {name} not found at {path}. Run Phase 3 training first."
            )
        model = joblib.load(path)
        base_models.append((name, model))
        print(f"  Loaded {name} from {path}")

    # Get feature-engineered data
    print(f"\n{'=' * 70}")
    print("  Preparing data")
    print("=" * 70)
    splits, _ = fit_and_transform(config_path)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    # Setup MLflow
    setup_mlflow(config)

    all_results = []

    # --- Individual model baselines (on test set) ---
    print(f"\n{'=' * 70}")
    print("  Individual Model Baselines (Test Set)")
    print("=" * 70)
    for name, model in base_models:
        y_proba = model.predict_proba(X_test)[:, 1]
        threshold = find_optimal_threshold(y_test.values, y_proba)
        metrics = compute_metrics(y_test.values, y_proba, threshold=threshold)
        metrics["model_name"] = name
        metrics["optimal_threshold"] = threshold
        all_results.append(metrics)
        print(f"  {name}: PR-AUC={metrics['pr_auc']:.4f}, F1={metrics['f1']:.4f}")

    # --- Stacking Ensemble ---
    print(f"\n{'=' * 70}")
    print("  Building Stacking Ensemble")
    print("=" * 70)
    stacking = StackingEnsemble(base_models, n_folds=5)
    stacking.fit(X_train, y_train)

    print(f"  Meta-learner weights: {stacking.base_weights_}")

    y_stack_proba = stacking.predict_proba(X_test)[:, 1]
    stack_threshold = find_optimal_threshold(y_test.values, y_stack_proba)
    stack_metrics = compute_metrics(
        y_test.values, y_stack_proba, threshold=stack_threshold
    )
    stack_metrics["model_name"] = "Stacking Ensemble"
    stack_metrics["optimal_threshold"] = stack_threshold
    all_results.append(stack_metrics)

    print(f"\n  Stacking Ensemble Results (Test Set):")
    print(f"    PR-AUC:    {stack_metrics['pr_auc']:.4f}")
    print(f"    ROC-AUC:   {stack_metrics['roc_auc']:.4f}")
    print(f"    F1:        {stack_metrics['f1']:.4f}")
    print(f"    Precision: {stack_metrics['precision']:.4f}")
    print(f"    Recall:    {stack_metrics['recall']:.4f}")

    # Log stacking to MLflow
    _log_ensemble_to_mlflow(
        "Stacking Ensemble",
        stacking,
        stack_metrics,
        stack_threshold,
        y_test,
        y_stack_proba,
        artifact_dir,
    )

    # Save stacking ensemble
    stacking.save(artifact_dir / "stacking_ensemble.joblib")
    print(f"  Saved to {artifact_dir / 'stacking_ensemble.joblib'}")

    # --- Blending Ensemble ---
    print(f"\n{'=' * 70}")
    print("  Building Blending Ensemble")
    print("=" * 70)
    blending = BlendingEnsemble(base_models)
    blending.fit(X_val, y_val, n_steps=21)

    y_blend_proba = blending.predict_proba(X_test)[:, 1]
    blend_threshold = find_optimal_threshold(y_test.values, y_blend_proba)
    blend_metrics = compute_metrics(
        y_test.values, y_blend_proba, threshold=blend_threshold
    )
    blend_metrics["model_name"] = "Blending Ensemble"
    blend_metrics["optimal_threshold"] = blend_threshold
    all_results.append(blend_metrics)

    print(f"\n  Blending Ensemble Results (Test Set):")
    print(f"    PR-AUC:    {blend_metrics['pr_auc']:.4f}")
    print(f"    ROC-AUC:   {blend_metrics['roc_auc']:.4f}")
    print(f"    F1:        {blend_metrics['f1']:.4f}")
    print(f"    Precision: {blend_metrics['precision']:.4f}")
    print(f"    Recall:    {blend_metrics['recall']:.4f}")

    # Log blending to MLflow
    _log_ensemble_to_mlflow(
        "Blending Ensemble",
        blending,
        blend_metrics,
        blend_threshold,
        y_test,
        y_blend_proba,
        artifact_dir,
    )

    # Save blending ensemble
    blending.save(artifact_dir / "blending_ensemble.joblib")
    print(f"  Saved to {artifact_dir / 'blending_ensemble.joblib'}")

    # --- Comparison ---
    print(f"\n\n{'=' * 70}")
    print("  FULL COMPARISON (Test Set)")
    print("=" * 70)
    print(format_results_table(all_results))

    # Determine best
    best = max(all_results, key=lambda r: r["pr_auc"])
    print(f"\n  Best model: {best['model_name']} (PR-AUC={best['pr_auc']:.4f})")

    # Save comparison results
    results_path = artifact_dir / "ensemble_results.json"
    serializable = [
        {k: v for k, v in r.items() if k != "threshold"} for r in all_results
    ]
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Results saved to {results_path}")

    return {
        "all_results": all_results,
        "stacking": stacking,
        "blending": blending,
        "best_model_name": best["model_name"],
    }


def _log_ensemble_to_mlflow(
    name: str,
    ensemble,
    metrics: dict,
    threshold: float,
    y_test,
    y_pred_proba: np.ndarray,
    artifact_dir: Path,
) -> None:
    """Log an ensemble's results to MLflow."""
    with mlflow.start_run(run_name=name):
        mlflow.log_metrics(
            {
                "test_pr_auc": metrics["pr_auc"],
                "test_roc_auc": metrics["roc_auc"],
                "test_f1": metrics["f1"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "optimal_threshold": threshold,
            }
        )

        y_true = np.asarray(y_test)
        y_pred = (y_pred_proba >= threshold).astype(int)

        safe_name = name.lower().replace(" ", "_")
        pr_path = artifact_dir / f"{safe_name}_pr_curve.png"
        plot_precision_recall_curve(y_true, y_pred_proba, name, save_path=pr_path)
        mlflow.log_artifact(str(pr_path))

        cm_path = artifact_dir / f"{safe_name}_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, name, save_path=cm_path)
        mlflow.log_artifact(str(cm_path))


if __name__ == "__main__":
    run_ensemble()
