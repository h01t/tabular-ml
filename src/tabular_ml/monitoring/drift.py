"""Data drift monitoring with Evidently."""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from evidently.presets import DataDriftPreset
from evidently import Report


def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_column: Optional[str] = None,
    numerical_features: Optional[list[str]] = None,
    categorical_features: Optional[list[str]] = None,
    threshold: float = 0.05,
    random_seed: int = 42,
) -> dict:
    """Detect data drift between reference and current datasets.

    Uses Evidently's DataDriftPreset to compute:
    - Feature-wise drift detection (statistical tests)
    - Dataset-level drift detection
    - Drift metrics for each feature

    Args:
        reference_data: Reference dataset (e.g., training data).
        current_data: Current/production dataset to monitor.
        target_column: Name of target column (if present). Optional.
        numerical_features: List of numerical feature column names.
        categorical_features: List of categorical feature column names.
        threshold: Significance level for drift detection (default 0.05).
        random_seed: Random seed for reproducibility.

    Returns:
        Dictionary containing drift report results.
    """
    if target_column is not None:
        # Remove target from features if present
        reference_features = reference_data.drop(columns=[target_column])
        current_features = current_data.drop(columns=[target_column])
    else:
        reference_features = reference_data
        current_features = current_data

    # Generate drift report
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    # Parse snapshot dict to extract drift results
    snapshot_dict = snapshot.dict()
    metrics = snapshot_dict.get("metrics", [])

    drifted_features = []
    dataset_drift = False
    drift_share = 0.0

    for metric in metrics:
        result = metric.get("result", {})
        # Look for drift_by_columns (feature-level drift)
        if "drift_by_columns" in result:
            drift_by_columns = result["drift_by_columns"]
            drifted_features = [
                col
                for col, drift_info in drift_by_columns.items()
                if drift_info.get("drift_detected", False)
            ]
            dataset_drift = len(drifted_features) > 0
            break
        # Look for drift_share (dataset-level drift)
        if "drift_share" in result:
            drift_share = result["drift_share"]
            dataset_drift = drift_share > threshold

    summary = {
        "dataset_drift_detected": dataset_drift,
        "drifted_features": drifted_features,
        "num_drifted_features": len(drifted_features),
        "total_features": len(reference_features.columns),
        "drift_threshold": threshold,
        "drift_share": drift_share,
        "reference_rows": len(reference_data),
        "current_rows": len(current_data),
    }

    # Optionally save HTML report
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_dir / "drift_detection_report.html"))

    return summary


def simulate_drift(
    data: pd.DataFrame,
    drift_type: str = "mean_shift",
    magnitude: float = 0.5,
    features_to_perturb: Optional[list[str]] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Simulate data drift by perturbing features.

    Useful for testing monitoring system.

    Args:
        data: Original dataset to perturb.
        drift_type: Type of drift to simulate:
            - "mean_shift": Shift feature means by magnitude * std
            - "scale_change": Scale feature values by (1 + magnitude)
            - "corruption": Add noise to features
            - "missing": Introduce missing values
        magnitude: Strength of the drift effect.
        features_to_perturb: List of feature names to perturb.
            If None, perturb all numerical features.
        random_seed: Random seed for reproducibility.

    Returns:
        DataFrame with simulated drift.
    """
    import numpy as np

    np.random.seed(random_seed)
    drifted_data = data.copy()

    # Identify numerical features if not specified
    if features_to_perturb is None:
        numerical_cols = drifted_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        # Exclude target if present
        if "Class" in numerical_cols:
            numerical_cols.remove("Class")
        features_to_perturb = numerical_cols

    # Apply drift based on type
    if drift_type == "mean_shift":
        for col in features_to_perturb:
            if col in drifted_data.columns:
                std = drifted_data[col].std()
                shift = magnitude * std
                drifted_data[col] = drifted_data[col] + shift

    elif drift_type == "scale_change":
        for col in features_to_perturb:
            if col in drifted_data.columns:
                drifted_data[col] = drifted_data[col] * (1 + magnitude)

    elif drift_type == "corruption":
        for col in features_to_perturb:
            if col in drifted_data.columns:
                noise = np.random.normal(
                    0, magnitude * drifted_data[col].std(), len(drifted_data)
                )
                drifted_data[col] = drifted_data[col] + noise

    elif drift_type == "missing":
        for col in features_to_perturb:
            if col in drifted_data.columns:
                mask = np.random.random(len(drifted_data)) < magnitude
                drifted_data.loc[mask, col] = np.nan

    else:
        raise ValueError(f"Unknown drift_type: {drift_type}")

    return drifted_data


def save_drift_report(
    report: dict,
    output_dir: Path | str = "artifacts",
    filename: str = "drift_report.json",
) -> Path:
    """Save drift report to disk as JSON.

    Args:
        report: Drift report dictionary.
        output_dir: Directory to save report.
        filename: Name of the report file.

    Returns:
        Path to saved report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return filepath


def load_drift_report(filepath: Path | str) -> dict:
    """Load drift report from disk.

    Args:
        filepath: Path to drift report JSON file.

    Returns:
        Drift report dictionary.
    """
    with open(filepath, "r") as f:
        return json.load(f)


def generate_drift_html_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_dir: Path | str = "artifacts",
    report_filename: str = "drift_report.html",
) -> Path:
    """Generate an HTML report showing data drift metrics.

    Creates an Evidently HTML report with data drift detection results.

    Args:
        reference_data: Reference dataset (e.g., training data).
        current_data: Current/production dataset to monitor.
        output_dir: Directory to save HTML report.
        report_filename: Name of the HTML file.

    Returns:
        Path to saved HTML report.
    """
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_data, current_data=current_data)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / report_filename
    snapshot.save_html(str(report_path))

    return report_path
