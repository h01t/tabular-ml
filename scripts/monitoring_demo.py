#!/usr/bin/env python
"""Demonstrate data drift monitoring on the fraud detection dataset.

Loads the trained preprocessing pipeline and transformed train/test splits,
then runs drift detection with Evidently.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tabular_ml.features.pipeline import load_pipeline
from tabular_ml.data.loader import load_data, split_data
from tabular_ml.monitoring.drift import (
    detect_data_drift,
    simulate_drift,
    generate_drift_html_report,
)


def main():
    print("=== Data Drift Monitoring Demo ===\n")

    # 1. Load raw data and split
    print("1. Loading dataset and splitting...")
    df = load_data()
    splits = split_data(df, stratify=True)
    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]

    print(f"   Train: {X_train.shape} rows, {X_train.shape[1]} features")
    print(f"   Test:  {X_test.shape} rows, {X_test.shape[1]} features")

    # 2. Load fitted preprocessing pipeline
    print("\n2. Loading preprocessing pipeline...")
    pipeline = load_pipeline()

    # Transform the raw splits using the pipeline
    print("   Transforming raw features...")
    X_train_transformed = pipeline.transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    print(f"   Transformed train: {X_train_transformed.shape[1]} features")
    print(f"   Transformed test:  {X_test_transformed.shape[1]} features")

    # 3. Detect drift between train and test (should be minimal)
    print("\n3. Detecting drift between train and test (no simulated drift)...")
    drift_result = detect_data_drift(
        reference_data=X_train_transformed,
        current_data=X_test_transformed,
        threshold=0.05,
    )

    print(f"   Dataset drift detected: {drift_result['dataset_drift_detected']}")
    print(f"   Drifted features: {drift_result['drifted_features']}")
    print(
        f"   Number of drifted features: {drift_result['num_drifted_features']}/{drift_result['total_features']}"
    )
    print(f"   Drift share: {drift_result.get('drift_share', 'N/A')}")

    # 4. Simulate drift on test data and detect
    print("\n4. Simulating drift on test data...")
    X_test_drifted = simulate_drift(
        data=X_test_transformed,
        drift_type="mean_shift",
        magnitude=2.0,  # Strong shift
        features_to_perturb=["V14", "V17"],  # Top correlated features
    )

    print("   Detecting drift between train and drifted test...")
    drift_result_sim = detect_data_drift(
        reference_data=X_train_transformed,
        current_data=X_test_drifted,
        threshold=0.05,
    )

    print(f"   Dataset drift detected: {drift_result_sim['dataset_drift_detected']}")
    print(f"   Drifted features: {drift_result_sim['drifted_features']}")
    print(
        f"   Number of drifted features: {drift_result_sim['num_drifted_features']}/{drift_result_sim['total_features']}"
    )

    # 5. Generate HTML reports
    print("\n5. Generating HTML reports...")
    # Report 1: Train vs Test (no drift)
    html_path1 = generate_drift_html_report(
        reference_data=X_train_transformed,
        current_data=X_test_transformed,
        output_dir="artifacts",
        report_filename="drift_report_train_vs_test.html",
    )
    print(f"   Generated: {html_path1}")

    # Report 2: Train vs Drifted Test
    html_path2 = generate_drift_html_report(
        reference_data=X_train_transformed,
        current_data=X_test_drifted,
        output_dir="artifacts",
        report_filename="drift_report_train_vs_drifted.html",
    )
    print(f"   Generated: {html_path2}")

    print("\n=== Demo Complete ===")
    print("Check the 'artifacts/' directory for HTML reports.")
    print("  - drift_report_train_vs_test.html: Baseline (minimal drift)")
    print("  - drift_report_train_vs_drifted.html: With simulated drift")


if __name__ == "__main__":
    main()
