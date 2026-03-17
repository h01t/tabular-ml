"""Tests for the FastAPI inference service."""

import joblib
import pytest
from httpx import ASGITransport, AsyncClient

from tabular_ml.api.app import FEATURE_COLUMNS, MODEL_PATH, PIPELINE_PATH, _state, app


@pytest.fixture(autouse=True)
def load_model_state():
    """Load model artifacts into app state before tests, clean up after."""
    _state["pipeline"] = joblib.load(PIPELINE_PATH)
    _state["model"] = joblib.load(MODEL_PATH)
    yield
    _state["pipeline"] = None
    _state["model"] = None


# ── Sample transaction data ──────────────────────────────────────────────

# A real legitimate transaction from the dataset (row 0)
LEGIT_TRANSACTION = {
    "Time": 0.0,
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    "V3": 2.53634673796914,
    "V4": 1.37815522427443,
    "V5": -0.338320769942518,
    "V6": 0.462387777762292,
    "V7": 0.239598554061257,
    "V8": 0.0986979012610507,
    "V9": 0.363786969611213,
    "V10": 0.0907941719789316,
    "V11": -0.551599533260813,
    "V12": -0.617800855762348,
    "V13": -0.991389847235408,
    "V14": -0.311169353699879,
    "V15": 1.46817697209427,
    "V16": -0.470400525259478,
    "V17": 0.207971241929242,
    "V18": 0.0257905801985591,
    "V19": 0.403992960255733,
    "V20": 0.251412098239705,
    "V21": -0.018306777944153,
    "V22": 0.277837575558899,
    "V23": -0.110473910188767,
    "V24": 0.0669280749146731,
    "V25": 0.128539358273528,
    "V26": -0.189114843888824,
    "V27": 0.133558376740387,
    "V28": -0.0210530534538215,
    "Amount": 149.62,
}

# A suspicious-looking transaction (exaggerated fraud-like features)
SUSPICIOUS_TRANSACTION = {
    "Time": 50000.0,
    "V1": -5.0,
    "V2": 3.0,
    "V3": -8.0,
    "V4": 5.0,
    "V5": -2.0,
    "V6": -3.0,
    "V7": -5.0,
    "V8": 1.0,
    "V9": -3.0,
    "V10": -8.0,
    "V11": 5.0,
    "V12": -10.0,
    "V13": 0.5,
    "V14": -12.0,
    "V15": 0.5,
    "V16": -8.0,
    "V17": -10.0,
    "V18": -5.0,
    "V19": 2.0,
    "V20": 1.0,
    "V21": 2.0,
    "V22": 0.5,
    "V23": -1.0,
    "V24": 0.5,
    "V25": 0.5,
    "V26": 0.5,
    "V27": 1.5,
    "V28": 0.5,
    "Amount": 500.0,
}


@pytest.fixture
def transport():
    return ASGITransport(app=app)


@pytest.mark.anyio
async def test_health_check(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["pipeline_loaded"] is True
    assert body["model_name"] == "XGBoost"
    assert body["model_version"] == "0.1.0"


@pytest.mark.anyio
async def test_predict_single_legit(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict", json=LEGIT_TRANSACTION)

    assert resp.status_code == 200
    body = resp.json()
    assert "is_fraud" in body
    assert "fraud_probability" in body
    assert "threshold" in body
    assert "model_name" in body
    assert 0 <= body["fraud_probability"] <= 1
    # This is a known legitimate transaction
    assert body["is_fraud"] is False


@pytest.mark.anyio
async def test_predict_single_suspicious(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict", json=SUSPICIOUS_TRANSACTION)

    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["fraud_probability"] <= 1
    # With extreme fraud-like features, probability should be high
    assert body["fraud_probability"] > 0.5


@pytest.mark.anyio
async def test_predict_batch(transport):
    batch = {"transactions": [LEGIT_TRANSACTION, SUSPICIOUS_TRANSACTION]}

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict/batch", json=batch)

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2

    # First should be legit, second should have higher fraud prob
    assert (
        body["predictions"][0]["fraud_probability"]
        < body["predictions"][1]["fraud_probability"]
    )


@pytest.mark.anyio
async def test_predict_batch_single_item(transport):
    batch = {"transactions": [LEGIT_TRANSACTION]}

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict/batch", json=batch)

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1


@pytest.mark.anyio
async def test_predict_missing_field(transport):
    incomplete = {k: v for k, v in LEGIT_TRANSACTION.items() if k != "Amount"}

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict", json=incomplete)

    assert resp.status_code == 422  # Pydantic validation error


@pytest.mark.anyio
async def test_predict_invalid_amount(transport):
    invalid = {**LEGIT_TRANSACTION, "Amount": -100}

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict", json=invalid)

    assert resp.status_code == 422  # Amount must be >= 0


@pytest.mark.anyio
async def test_predict_batch_empty(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/predict/batch", json={"transactions": []})

    assert resp.status_code == 422  # min_length=1


@pytest.mark.anyio
async def test_predict_response_consistency(transport):
    """Single predict and batch predict should return identical results for same input."""
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        single_resp = await client.post("/predict", json=LEGIT_TRANSACTION)
        batch_resp = await client.post(
            "/predict/batch", json={"transactions": [LEGIT_TRANSACTION]}
        )

    single = single_resp.json()
    batch_first = batch_resp.json()["predictions"][0]

    assert single["fraud_probability"] == batch_first["fraud_probability"]
    assert single["is_fraud"] == batch_first["is_fraud"]
