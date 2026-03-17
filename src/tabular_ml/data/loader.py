"""Data loading and train/val/test splitting for credit card fraud detection."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str | Path = "data/raw/creditcard.csv") -> pd.DataFrame:
    """Load the credit card fraud dataset from CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with all columns.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See README for download instructions."
        )
    df = pd.read_csv(path)
    return df


def split_data(
    df: pd.DataFrame,
    target_column: str = "Class",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Split data into train/val/test sets with stratification.

    Uses a two-step split:
    1. Split off the test set from the full data.
    2. Split the remaining data into train and validation sets.

    Args:
        df: Full dataset.
        target_column: Name of the target column.
        test_size: Fraction of full data for the test set.
        val_size: Fraction of the remaining (non-test) data for validation.
        random_state: Random seed for reproducibility.
        stratify: Whether to stratify splits by target class.

    Returns:
        Dictionary with keys "train", "val", "test", each mapping to
        a tuple of (X, y) where X is a DataFrame and y is a Series.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify_col = y if stratify else None

    # Step 1: split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    # Step 2: split remaining into train and val
    stratify_temp = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_temp,
    )

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    # Log split info
    for name, (X_split, y_split) in splits.items():
        n_fraud = y_split.sum()
        pct = n_fraud / len(y_split) * 100
        print(f"  {name:5s}: {len(X_split):>7,} rows | {n_fraud:>3} fraud ({pct:.3f}%)")

    return splits
