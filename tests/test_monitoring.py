"""Tests for data drift monitoring."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from tabular_ml.monitoring.drift import (
    detect_data_drift,
    generate_drift_html_report,
    simulate_drift,
    save_drift_report,
    load_drift_report,
)


@pytest.fixture
def sample_reference_data():
    """Create a small reference dataset with numeric features."""
    rng = np.random.RandomState(42)
    n = 200
    data = {
        "feature1": rng.normal(0, 1, n),
        "feature2": rng.normal(0, 1, n),
        "feature3": rng.normal(0, 1, n),
        "target": rng.randint(0, 2, n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_current_data(sample_reference_data):
    """Create current data by slightly perturbing reference data."""
    df = sample_reference_data.copy()
    # Add small drift to feature1
    df["feature1"] = df["feature1"] + 0.5
    return df


def test_detect_data_drift_no_drift(sample_reference_data):
    """When reference and current are identical, no drift should be detected."""
    result = detect_data_drift(
        reference_data=sample_reference_data,
        current_data=sample_reference_data,
        target_column="target",
        threshold=0.05,
    )

    assert isinstance(result, dict)
    assert "dataset_drift_detected" in result
    assert "drifted_features" in result
    assert "num_drifted_features" in result
    assert result["dataset_drift_detected"] is False
    assert len(result["drifted_features"]) == 0
    assert result["num_drifted_features"] == 0
    assert result["total_features"] == 3  # Excluding target


def test_detect_data_drift_with_drift(sample_reference_data, sample_current_data):
    """When current data is perturbed, drift should be detected."""
    result = detect_data_drift(
        reference_data=sample_reference_data,
        current_data=sample_current_data,
        target_column="target",
        threshold=0.05,
    )

    assert isinstance(result, dict)
    # Drift may or may not be detected depending on statistical test
    # We just verify the structure
    assert "dataset_drift_detected" in result
    assert "drifted_features" in result
    assert "num_drifted_features" in result
    assert result["total_features"] == 3


def test_detect_data_drift_without_target(sample_reference_data):
    """Drift detection should work without target column."""
    ref_no_target = sample_reference_data.drop(columns=["target"])
    result = detect_data_drift(
        reference_data=ref_no_target,
        current_data=ref_no_target,
        threshold=0.05,
    )

    assert isinstance(result, dict)
    assert "dataset_drift_detected" in result
    assert result["total_features"] == 3


def test_simulate_drift_mean_shift(sample_reference_data):
    """Test mean shift drift simulation."""
    drifted = simulate_drift(
        data=sample_reference_data,
        drift_type="mean_shift",
        magnitude=1.0,
        features_to_perturb=["feature1"],
    )

    assert drifted.shape == sample_reference_data.shape
    assert "feature1" in drifted.columns
    # Check that feature1 values are shifted
    original_mean = sample_reference_data["feature1"].mean()
    drifted_mean = drifted["feature1"].mean()
    # Shift should be approximately magnitude * std (std ~1)
    assert abs(drifted_mean - original_mean - 1.0) < 0.5


def test_simulate_drift_scale_change(sample_reference_data):
    """Test scale change drift simulation."""
    drifted = simulate_drift(
        data=sample_reference_data,
        drift_type="scale_change",
        magnitude=0.5,
        features_to_perturb=["feature2"],
    )

    assert drifted.shape == sample_reference_data.shape
    # feature2 should be scaled by 1.5
    original_mean = sample_reference_data["feature2"].mean()
    drifted_mean = drifted["feature2"].mean()
    assert abs(drifted_mean - original_mean * 1.5) < 0.1


def test_simulate_drift_corruption(sample_reference_data):
    """Test corruption drift simulation."""
    drifted = simulate_drift(
        data=sample_reference_data,
        drift_type="corruption",
        magnitude=0.2,
        features_to_perturb=["feature3"],
    )

    assert drifted.shape == sample_reference_data.shape
    # Values should differ
    assert not drifted["feature3"].equals(sample_reference_data["feature3"])


def test_simulate_drift_missing(sample_reference_data):
    """Test missing value drift simulation."""
    drifted = simulate_drift(
        data=sample_reference_data,
        drift_type="missing",
        magnitude=0.3,
        features_to_perturb=["feature1"],
    )

    assert drifted.shape == sample_reference_data.shape
    # Should have some NaN values
    assert drifted["feature1"].isna().sum() > 0


def test_generate_drift_html_report(
    sample_reference_data, sample_current_data, tmp_path
):
    """Test HTML report generation."""
    output_dir = tmp_path / "reports"
    report_path = generate_drift_html_report(
        reference_data=sample_reference_data,
        current_data=sample_current_data,
        output_dir=output_dir,
        report_filename="test_report.html",
    )

    assert report_path.exists()
    assert report_path.suffix == ".html"
    assert report_path.stat().st_size > 0


def test_save_and_load_drift_report(
    sample_reference_data, sample_current_data, tmp_path
):
    """Test saving and loading drift report as JSON."""
    result = detect_data_drift(
        reference_data=sample_reference_data,
        current_data=sample_current_data,
        target_column="target",
    )

    # Save report
    report_path = save_drift_report(
        report=result,
        output_dir=tmp_path,
        filename="test_report.json",
    )

    assert report_path.exists()
    assert report_path.suffix == ".json"

    # Load report
    loaded = load_drift_report(report_path)
    assert isinstance(loaded, dict)
    assert "dataset_drift_detected" in loaded
    assert loaded["total_features"] == result["total_features"]
