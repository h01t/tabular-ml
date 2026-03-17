"""Feature engineering transformers — sklearn-compatible pipeline components."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract cyclical hour-of-day features from the Time column.

    The Time column represents seconds elapsed from the first transaction.
    We convert to hour-of-day and encode as sin/cos for cyclical continuity
    (so hour 23 is close to hour 0).
    """

    def __init__(
        self,
        time_column: str = "Time",
        period_seconds: int = 86400,
        drop_original: bool = True,
    ):
        self.time_column = time_column
        self.period_seconds = period_seconds
        self.drop_original = drop_original

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Convert seconds to hour-of-day (0-24 range)
        hours = (X[self.time_column] % self.period_seconds) / 3600.0

        # Cyclical encoding
        X["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        X["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

        if self.drop_original:
            X = X.drop(columns=[self.time_column])

        return X


class AmountTransformer(BaseEstimator, TransformerMixin):
    """Transform the Amount feature: log-transform and standardize.

    Applies log1p transform to handle the heavy right skew observed in EDA,
    then standardizes to zero mean and unit variance.
    """

    def __init__(
        self,
        amount_column: str = "Amount",
        log_transform: bool = True,
        standardize: bool = True,
    ):
        self.amount_column = amount_column
        self.log_transform = log_transform
        self.standardize = standardize
        self._scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y=None):
        values = X[[self.amount_column]].copy()
        if self.log_transform:
            values = np.log1p(values)
        if self.standardize:
            self._scaler.fit(values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        values = X[[self.amount_column]].copy()

        if self.log_transform:
            values = np.log1p(values)
        if self.standardize:
            values = pd.DataFrame(
                self._scaler.transform(values),
                columns=[self.amount_column],
                index=X.index,
            )

        X[self.amount_column] = values[self.amount_column]
        return X


class InteractionFeatureCreator(BaseEstimator, TransformerMixin):
    """Create interaction features (product) between specified feature pairs.

    Based on EDA findings, combinations of top fraud-correlated V-features
    can capture non-linear patterns.
    """

    def __init__(self, pairs: list[list[str]] | None = None):
        self.pairs = pairs or [
            ["V14", "V17"],
            ["V12", "V14"],
            ["V10", "V17"],
            ["V4", "V11"],
        ]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for a, b in self.pairs:
            col_name = f"{a}_x_{b}"
            X[col_name] = X[a] * X[b]
        return X


def build_feature_pipeline(
    time_period_seconds: int = 86400,
    amount_log: bool = True,
    amount_standardize: bool = True,
    interaction_pairs: list[list[str]] | None = None,
    drop_time: bool = True,
) -> Pipeline:
    """Build the full feature engineering pipeline.

    Pipeline order:
    1. Extract cyclical time features (and optionally drop raw Time)
    2. Transform Amount (log + standardize)
    3. Create interaction features

    Args:
        time_period_seconds: Period for hour-of-day extraction (default 86400 = 1 day).
        amount_log: Whether to log-transform Amount.
        amount_standardize: Whether to standardize Amount.
        interaction_pairs: List of [feature_a, feature_b] pairs for interactions.
        drop_time: Whether to drop the original Time column.

    Returns:
        Fitted-ready sklearn Pipeline.
    """
    return Pipeline(
        [
            (
                "time_features",
                TimeFeatureExtractor(
                    period_seconds=time_period_seconds,
                    drop_original=drop_time,
                ),
            ),
            (
                "amount_transform",
                AmountTransformer(
                    log_transform=amount_log,
                    standardize=amount_standardize,
                ),
            ),
            (
                "interactions",
                InteractionFeatureCreator(
                    pairs=interaction_pairs,
                ),
            ),
        ]
    )
