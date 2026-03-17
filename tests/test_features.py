"""Tests for feature engineering transformers and pipeline."""

import numpy as np
import pandas as pd
import pytest

from tabular_ml.features.engineering import (
    AmountTransformer,
    InteractionFeatureCreator,
    TimeFeatureExtractor,
    build_feature_pipeline,
)


@pytest.fixture
def sample_X():
    """Create a small feature DataFrame mimicking the fraud dataset (without target)."""
    rng = np.random.RandomState(42)
    n = 200
    data = {
        "Time": rng.uniform(0, 172792, n),
        "Amount": rng.exponential(88, n),
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.randn(n)
    return pd.DataFrame(data)


class TestTimeFeatureExtractor:
    def test_adds_sin_cos_columns(self, sample_X):
        transformer = TimeFeatureExtractor(drop_original=False)
        result = transformer.fit_transform(sample_X)
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "Time" in result.columns  # kept

    def test_drops_time_column(self, sample_X):
        transformer = TimeFeatureExtractor(drop_original=True)
        result = transformer.fit_transform(sample_X)
        assert "Time" not in result.columns
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns

    def test_sin_cos_range(self, sample_X):
        transformer = TimeFeatureExtractor()
        result = transformer.fit_transform(sample_X)
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()

    def test_cyclical_continuity(self):
        """Hour 0 and hour 24 should produce the same sin/cos values."""
        df = pd.DataFrame(
            {
                "Time": [0.0, 86400.0],  # exactly one day apart
                "Amount": [10.0, 10.0],
            }
        )
        for i in range(1, 29):
            df[f"V{i}"] = 0.0

        transformer = TimeFeatureExtractor()
        result = transformer.fit_transform(df)
        assert np.isclose(
            result["hour_sin"].iloc[0], result["hour_sin"].iloc[1], atol=1e-10
        )
        assert np.isclose(
            result["hour_cos"].iloc[0], result["hour_cos"].iloc[1], atol=1e-10
        )


class TestAmountTransformer:
    def test_log_transform_applied(self, sample_X):
        transformer = AmountTransformer(log_transform=True, standardize=False)
        result = transformer.fit_transform(sample_X)
        # log1p(0) = 0, and log1p(x) < x for x > 0
        expected = np.log1p(sample_X["Amount"])
        np.testing.assert_array_almost_equal(result["Amount"].values, expected.values)

    def test_standardized_mean_near_zero(self, sample_X):
        transformer = AmountTransformer(log_transform=True, standardize=True)
        result = transformer.fit_transform(sample_X)
        assert abs(result["Amount"].mean()) < 0.1  # close to 0
        assert abs(result["Amount"].std() - 1.0) < 0.1  # close to 1

    def test_does_not_alter_other_columns(self, sample_X):
        transformer = AmountTransformer()
        result = transformer.fit_transform(sample_X)
        # V1 should be unchanged
        pd.testing.assert_series_equal(result["V1"], sample_X["V1"])

    def test_fit_transform_consistency(self, sample_X):
        """Transform on train data should match fit_transform."""
        transformer = AmountTransformer()
        result_fit_transform = transformer.fit_transform(sample_X)
        result_transform = transformer.transform(sample_X)
        pd.testing.assert_frame_equal(result_fit_transform, result_transform)


class TestInteractionFeatureCreator:
    def test_creates_interaction_columns(self, sample_X):
        pairs = [["V14", "V17"], ["V4", "V11"]]
        transformer = InteractionFeatureCreator(pairs=pairs)
        result = transformer.fit_transform(sample_X)
        assert "V14_x_V17" in result.columns
        assert "V4_x_V11" in result.columns

    def test_interaction_values_correct(self, sample_X):
        pairs = [["V14", "V17"]]
        transformer = InteractionFeatureCreator(pairs=pairs)
        result = transformer.fit_transform(sample_X)
        expected = sample_X["V14"] * sample_X["V17"]
        np.testing.assert_array_almost_equal(
            result["V14_x_V17"].values, expected.values
        )

    def test_original_columns_preserved(self, sample_X):
        pairs = [["V14", "V17"]]
        transformer = InteractionFeatureCreator(pairs=pairs)
        result = transformer.fit_transform(sample_X)
        assert "V14" in result.columns
        assert "V17" in result.columns

    def test_default_pairs(self, sample_X):
        transformer = InteractionFeatureCreator()
        result = transformer.fit_transform(sample_X)
        # Default: 4 pairs
        expected_new = {"V14_x_V17", "V12_x_V14", "V10_x_V17", "V4_x_V11"}
        new_cols = set(result.columns) - set(sample_X.columns)
        assert new_cols == expected_new


class TestBuildFeaturePipeline:
    def test_full_pipeline_output_shape(self, sample_X):
        pipeline = build_feature_pipeline()
        result = pipeline.fit_transform(sample_X)

        # Original: 30 cols (Time + Amount + V1-V28)
        # After: Time dropped (-1), hour_sin/hour_cos added (+2), 4 interactions (+4)
        # Net: 30 - 1 + 2 + 4 = 35
        assert result.shape == (len(sample_X), 35)

    def test_pipeline_is_idempotent_on_transform(self, sample_X):
        """Calling transform multiple times gives the same result."""
        pipeline = build_feature_pipeline()
        pipeline.fit(sample_X)
        result1 = pipeline.transform(sample_X)
        result2 = pipeline.transform(sample_X)
        pd.testing.assert_frame_equal(result1, result2)

    def test_pipeline_no_time_column(self, sample_X):
        pipeline = build_feature_pipeline(drop_time=True)
        result = pipeline.fit_transform(sample_X)
        assert "Time" not in result.columns

    def test_pipeline_generalizes_to_new_data(self, sample_X):
        """Pipeline fitted on one set transforms a different set correctly."""
        pipeline = build_feature_pipeline()
        # Fit on first half
        pipeline.fit(sample_X.iloc[:100])
        # Transform second half
        result = pipeline.transform(sample_X.iloc[100:])
        assert result.shape[0] == 100
        assert "hour_sin" in result.columns
