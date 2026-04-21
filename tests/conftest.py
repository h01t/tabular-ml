"""Shared test configuration."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    """Restrict async tests to asyncio only (skip trio)."""
    return request.param


def optional_dependency_available(module_name: str) -> bool:
    """Return True when an optional dependency imports successfully."""
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


class IdentityPipeline:
    """Minimal pipeline stub for API tests."""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()


class HeuristicFraudModel:
    """Small deterministic model stub for API tests.

    Produces low scores for benign samples and high scores for obviously
    suspicious combinations of high amount and large-magnitude V-features.
    """

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        risk = (
            np.abs(X["V14"])
            + np.abs(X["V17"])
            + np.abs(X["V10"])
            + np.clip(X["V4"], 0, None)
            + X["Amount"] / 100
        )
        fraud_prob = np.clip(risk / 10, 0, 1).to_numpy()
        return np.column_stack([1 - fraud_prob, fraud_prob])
