"""Orchestration script — tune and train all models, produce results comparison."""

import json
from pathlib import Path

from tabular_ml.features.pipeline import fit_and_transform, load_config, save_pipeline
from tabular_ml.models.evaluation import (
    compute_metrics,
    find_optimal_threshold,
    format_results_table,
)
from tabular_ml.models.trainer import (
    compute_class_weight,
    setup_mlflow,
    train_and_evaluate,
)
from tabular_ml.models.tuning import build_model, tune_model


MODEL_NAMES = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}


def run_full_training(config_path: str = "configs/default.yaml") -> list[dict]:
    """Run the complete training pipeline: feature engineering → tuning → training → evaluation.

    Steps for each model:
    1. Optuna hyperparameter search (optimizing PR-AUC on validation set)
    2. Train final model with best params
    3. Evaluate and log to MLflow
    4. Evaluate on held-out test set

    Args:
        config_path: Path to YAML config.

    Returns:
        List of result dicts for each model.
    """
    config = load_config(config_path)
    artifact_dir = Path(config["pipeline"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Feature engineering
    print("=" * 70)
    print("  STEP 1: Feature Engineering")
    print("=" * 70)
    splits, feat_pipeline = fit_and_transform(config_path)
    save_pipeline(feat_pipeline, config_path)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    # Setup MLflow
    setup_mlflow(config)

    all_results = []

    # Tune and train each model
    for model_type, display_name in MODEL_NAMES.items():
        print(f"\n{'=' * 70}")
        print(f"  STEP 2: Tuning {display_name}")
        print(f"{'=' * 70}")

        # Hyperparameter tuning
        tuning_result = tune_model(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            config=config,
        )

        # Train final model with best params
        print(f"\n  Training final {display_name} with best params...")
        best_params = tuning_result["best_params"]
        model = build_model(
            model_type,
            best_params,
            config=config,
            backend=tuning_result["backend"],
        )

        result = train_and_evaluate(
            model=model,
            model_name=display_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params=best_params,
            config=config,
        )

        # Also evaluate on test set
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_threshold = find_optimal_threshold(y_test.values, y_test_proba)
        test_metrics = compute_metrics(
            y_test.values, y_test_proba, threshold=test_threshold
        )
        result["test_metrics"] = test_metrics

        print(
            f"\n  {display_name} Test Set: PR-AUC={test_metrics['pr_auc']:.4f}, "
            f"F1={test_metrics['f1']:.4f}, Recall={test_metrics['recall']:.4f}"
        )

        all_results.append(result)

    # Print comparison table
    print(f"\n\n{'=' * 70}")
    print("  RESULTS COMPARISON (Validation Set)")
    print(f"{'=' * 70}")
    print(format_results_table(all_results))

    print(f"\n{'=' * 70}")
    print("  RESULTS COMPARISON (Test Set)")
    print(f"{'=' * 70}")
    test_results = [
        {"model_name": r["model_name"], **r["test_metrics"]} for r in all_results
    ]
    print(format_results_table(test_results))

    # Save results to JSON
    results_path = artifact_dir / "training_results.json"
    serializable = []
    for r in all_results:
        entry = {
            "model_name": r["model_name"],
            "val_pr_auc": r["pr_auc"],
            "val_roc_auc": r["roc_auc"],
            "val_f1": r["f1"],
            "val_precision": r["precision"],
            "val_recall": r["recall"],
            "optimal_threshold": r["optimal_threshold"],
            "test_pr_auc": r["test_metrics"]["pr_auc"],
            "test_roc_auc": r["test_metrics"]["roc_auc"],
            "test_f1": r["test_metrics"]["f1"],
            "test_precision": r["test_metrics"]["precision"],
            "test_recall": r["test_metrics"]["recall"],
        }
        serializable.append(entry)

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    run_full_training()
