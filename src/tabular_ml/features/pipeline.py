"""Pipeline orchestration — fit, transform, save, and load the feature pipeline."""

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from tabular_ml.data.loader import load_data, split_data
from tabular_ml.features.engineering import build_feature_pipeline


def load_config(config_path: str | Path = "configs/default.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def fit_and_transform(
    config_path: str | Path = "configs/default.yaml",
) -> tuple[dict[str, tuple[pd.DataFrame, pd.Series]], Pipeline]:
    """Full pipeline: load data, split, fit feature pipeline on train, transform all splits.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Tuple of (splits_dict, fitted_pipeline) where splits_dict maps
        "train"/"val"/"test" to (X_transformed, y) tuples.
    """
    config = load_config(config_path)

    # Load and split
    print("Loading data...")
    df = load_data(config["data"]["raw_path"])

    print("Splitting data...")
    splits = split_data(
        df,
        target_column=config["data"]["target_column"],
        test_size=config["split"]["test_size"],
        val_size=config["split"]["val_size"],
        random_state=config["split"]["random_state"],
        stratify=config["split"]["stratify"],
    )

    # Build pipeline
    feat_cfg = config["features"]
    pipeline = build_feature_pipeline(
        time_period_seconds=feat_cfg["time_period_seconds"],
        amount_log=feat_cfg["amount_log_transform"],
        amount_standardize=feat_cfg["amount_standardize"],
        interaction_pairs=feat_cfg["interaction_pairs"],
        drop_time="Time" in feat_cfg.get("drop_original", []),
    )

    # Fit on train only, transform all splits
    X_train, y_train = splits["train"]
    print("Fitting feature pipeline on training data...")
    pipeline.fit(X_train)

    transformed_splits = {}
    for name, (X, y) in splits.items():
        X_transformed = pipeline.transform(X)
        transformed_splits[name] = (X_transformed, y)
        print(f"  {name:5s}: {X_transformed.shape[1]} features after engineering")

    return transformed_splits, pipeline


def save_pipeline(
    pipeline: Pipeline, config_path: str | Path = "configs/default.yaml"
) -> Path:
    """Serialize the fitted pipeline to disk.

    Args:
        pipeline: Fitted sklearn Pipeline.
        config_path: Path to config for artifact directory settings.

    Returns:
        Path to the saved pipeline file.
    """
    config = load_config(config_path)
    artifact_dir = Path(config["pipeline"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    filepath = artifact_dir / config["pipeline"]["pipeline_filename"]
    joblib.dump(pipeline, filepath)
    print(f"Pipeline saved to {filepath}")
    return filepath


def load_pipeline(config_path: str | Path = "configs/default.yaml") -> Pipeline:
    """Load a previously saved pipeline from disk.

    Args:
        config_path: Path to config for artifact directory settings.

    Returns:
        Fitted sklearn Pipeline.
    """
    config = load_config(config_path)
    filepath = (
        Path(config["pipeline"]["artifact_dir"])
        / config["pipeline"]["pipeline_filename"]
    )

    if not filepath.exists():
        raise FileNotFoundError(
            f"No saved pipeline found at {filepath}. Run fit_and_transform first."
        )

    pipeline = joblib.load(filepath)
    print(f"Pipeline loaded from {filepath}")
    return pipeline
