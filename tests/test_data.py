"""Tests for data loading and splitting."""

import numpy as np
import pandas as pd
import pytest

from tabular_ml.data.loader import load_data, split_data


@pytest.fixture
def sample_df():
    """Create a small synthetic dataset mimicking the fraud dataset structure."""
    rng = np.random.RandomState(42)
    n = 1000
    data = {
        "Time": rng.uniform(0, 172792, n),
        "Amount": rng.exponential(88, n),
        "Class": np.concatenate([np.zeros(990, dtype=int), np.ones(10, dtype=int)]),
    }
    # Add V1-V28
    for i in range(1, 29):
        data[f"V{i}"] = rng.randn(n)
    return pd.DataFrame(data)


class TestLoadData:
    def test_load_csv_from_path(self, sample_df, tmp_path):
        """Load a CSV from disk and preserve its columns and row count."""
        csv_path = tmp_path / "creditcard.csv"
        sample_df.to_csv(csv_path, index=False)

        df = load_data(csv_path)
        assert df.shape == sample_df.shape
        assert list(df.columns) == list(sample_df.columns)

    def test_load_missing_file_raises(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_data("data/raw/nonexistent.csv")


class TestSplitData:
    def test_split_sizes(self, sample_df):
        """Verify split proportions are approximately correct."""
        splits = split_data(sample_df, test_size=0.2, val_size=0.2)

        n_total = len(sample_df)
        n_test = len(splits["test"][0])
        n_val = len(splits["val"][0])
        n_train = len(splits["train"][0])

        # All rows accounted for
        assert n_train + n_val + n_test == n_total

        # Approximate proportions (allow 5% tolerance)
        assert abs(n_test / n_total - 0.2) < 0.05
        assert abs(n_val / n_total - 0.16) < 0.05  # 0.2 * 0.8 = 0.16

    def test_no_data_leakage(self, sample_df):
        """Verify no index overlap between splits."""
        splits = split_data(sample_df, test_size=0.2, val_size=0.2)

        train_idx = set(splits["train"][0].index)
        val_idx = set(splits["val"][0].index)
        test_idx = set(splits["test"][0].index)

        assert len(train_idx & val_idx) == 0, "Train/val overlap"
        assert len(train_idx & test_idx) == 0, "Train/test overlap"
        assert len(val_idx & test_idx) == 0, "Val/test overlap"

    def test_stratification_preserves_class_ratio(self, sample_df):
        """Fraud rate should be roughly equal across all splits."""
        splits = split_data(sample_df, test_size=0.2, val_size=0.2, stratify=True)

        original_rate = sample_df["Class"].mean()
        for name, (X, y) in splits.items():
            split_rate = y.mean()
            # Allow 2% absolute tolerance (small dataset)
            assert abs(split_rate - original_rate) < 0.02, (
                f"{name} fraud rate {split_rate:.4f} differs from original {original_rate:.4f}"
            )

    def test_target_separated_from_features(self, sample_df):
        """Target column should not appear in X."""
        splits = split_data(sample_df)
        for name, (X, y) in splits.items():
            assert "Class" not in X.columns, f"Target in {name} features"
            assert y.name == "Class"
