"""FastAPI inference service for credit card fraud detection."""

from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from tabular_ml.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
    TransactionFeatures,
)
from tabular_ml.config import get_serving_settings, load_config

# ── Feature column order (must match training) ──────────────────────────
FEATURE_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

# ── Global state populated at startup ────────────────────────────────────
_serving_settings = get_serving_settings(load_config())

_state: dict = {
    "pipeline": None,
    "model": None,
    "model_name": _serving_settings["model_name"],
    "model_version": _serving_settings["model_version"],
    "threshold": _serving_settings["threshold"],
}

# ── Artifact paths (configurable via env in production) ──────────────────
PIPELINE_PATH = _serving_settings["pipeline_path"]
MODEL_PATH = _serving_settings["model_path"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup, clean up on shutdown."""
    # Startup
    if not PIPELINE_PATH.exists():
        raise RuntimeError(f"Preprocessing pipeline not found at {PIPELINE_PATH}")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")

    _state["pipeline"] = joblib.load(PIPELINE_PATH)
    _state["model"] = joblib.load(MODEL_PATH)
    print(f"Loaded pipeline from {PIPELINE_PATH}")
    print(f"Loaded model from {MODEL_PATH}")

    yield

    # Shutdown
    _state["pipeline"] = None
    _state["model"] = None


app = FastAPI(
    title=_serving_settings["api_title"],
    description=_serving_settings["api_description"],
    version=_serving_settings["api_version"],
    lifespan=lifespan,
)


def _transaction_to_dataframe(transaction: TransactionFeatures) -> pd.DataFrame:
    """Convert a single TransactionFeatures to a 1-row DataFrame."""
    data = transaction.model_dump()
    return pd.DataFrame([data], columns=FEATURE_COLUMNS)


def _transactions_to_dataframe(transactions: list[TransactionFeatures]) -> pd.DataFrame:
    """Convert a list of TransactionFeatures to a DataFrame."""
    records = [t.model_dump() for t in transactions]
    return pd.DataFrame(records, columns=FEATURE_COLUMNS)


def _predict_single(df: pd.DataFrame) -> tuple[float, bool]:
    """Run preprocessing + model prediction on a single-row DataFrame.

    Returns:
        Tuple of (fraud_probability, is_fraud).
    """
    pipeline = _state["pipeline"]
    model = _state["model"]
    threshold = _state["threshold"]

    X_transformed = pipeline.transform(df)
    proba = model.predict_proba(X_transformed)[:, 1]
    fraud_prob = float(proba[0])
    is_fraud = fraud_prob >= threshold

    return fraud_prob, is_fraud


def _predict_batch(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Run preprocessing + model prediction on a multi-row DataFrame.

    Returns:
        Tuple of (fraud_probabilities, is_fraud_array).
    """
    pipeline = _state["pipeline"]
    model = _state["model"]
    threshold = _state["threshold"]

    X_transformed = pipeline.transform(df)
    probas = model.predict_proba(X_transformed)[:, 1]
    is_fraud = probas >= threshold

    return probas, is_fraud


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — confirms the service is running and artifacts are loaded."""
    return HealthResponse(
        status="healthy",
        model_loaded=_state["model"] is not None,
        pipeline_loaded=_state["pipeline"] is not None,
        model_name=_state["model_name"],
        model_version=_state["model_version"],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionFeatures):
    """Predict fraud probability for a single transaction.

    Accepts raw transaction features (Time, V1-V28, Amount),
    applies the preprocessing pipeline, and returns the fraud prediction.
    """
    if _state["model"] is None or _state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = _transaction_to_dataframe(transaction)
    fraud_prob, is_fraud = _predict_single(df)

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(fraud_prob, 6),
        threshold=_state["threshold"],
        model_name=_state["model_name"],
        model_version=_state["model_version"],
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict fraud probability for a batch of transactions (up to 10,000).

    More efficient than calling /predict in a loop — processes all
    transactions in a single vectorized pass.
    """
    if _state["model"] is None or _state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = _transactions_to_dataframe(request.transactions)
    probas, is_fraud = _predict_batch(df)

    predictions = [
        PredictionResponse(
            is_fraud=bool(is_fraud[i]),
            fraud_probability=round(float(probas[i]), 6),
            threshold=_state["threshold"],
            model_name=_state["model_name"],
            model_version=_state["model_version"],
        )
        for i in range(len(probas))
    ]

    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
    )
