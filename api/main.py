"""
FastAPI application for the Fraud Detection System.

Endpoints:
    POST /predict       — Score a single transaction
    POST /predict/batch — Score multiple transactions
    GET  /health        — Health check
    GET  /model/info    — Model metadata
"""

import time
import threading
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# In-memory sliding window cache for behavioral velocity checks
# Stores mapping: card_id -> list of {"timestamp": float, "amount": float}
VELOCITY_CACHE = defaultdict(list)
VELOCITY_LOCK = threading.Lock()
VELOCITY_WINDOW_SECONDS = 600  # 10 minutes sliding window

from api.dependencies import load_explainer, load_model, prepare_features, load_test_df
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionInput,
)
from src.config import ENGINEERED_FEATURES
from src.explain import shap_values_to_api_response
from src.utils import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Application Setup
# ──────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Production-grade API for credit card fraud detection. "
        "Returns fraud probability, binary classification, and "
        "SHAP-based explainability for every prediction."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request Logging Middleware
# ──────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request method, path, and latency for every request."""
    start_time = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start_time) * 1000

    logger.info(
        f"{request.method} {request.url.path} — "
        f"status={response.status_code} latency={latency_ms:.1f}ms"
    )
    response.headers["X-Latency-Ms"] = f"{latency_ms:.1f}"
    return response


# ──────────────────────────────────────────────
# Startup Event
# ──────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Pre-load model and explainer on startup for fast first request."""
    try:
        load_model()
        load_explainer()
        logger.info("Model and SHAP explainer loaded successfully on startup ✓")
    except FileNotFoundError as e:
        logger.warning(f"Model not found on startup: {e}")
        logger.warning("Endpoints will fail until a model is trained and saved.")


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
def check_velocity_and_record(card_id: str, amount: float) -> tuple[bool, list[str]]:
    """
    Check if a transaction violates the card velocity checks,
    and record the current transaction in history.
    """
    now = time.time()
    triggered = False
    reasons = []

    with VELOCITY_LOCK:
        # Get history and filter out old transactions
        history = VELOCITY_CACHE[card_id]
        active_history = [
            tx for tx in history
            if now - tx["timestamp"] <= VELOCITY_WINDOW_SECONDS
        ]

        # Rule 2: Exceeded 3 transactions in 10 minutes sliding window
        if len(active_history) >= 3:
            triggered = True
            reasons.append(
                f"Velocity Alert: Card exceeded 3 transactions in 10 minutes "
                f"({len(active_history) + 1} attempts detected)"
            )

        # Rule 3: Exceeded $1,000 spending in 10 minutes sliding window
        total_spent = sum(tx["amount"] for tx in active_history)
        if total_spent + amount > 1000.0:
            triggered = True
            reasons.append(
                f"Velocity Alert: Card spending exceeded $1,000 in 10 minutes "
                f"(Total: ${total_spent + amount:.2f})"
            )

        # Record this transaction in history
        active_history.append({"timestamp": now, "amount": amount})
        VELOCITY_CACHE[card_id] = active_history

    return triggered, reasons


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionInput):
    """
    Score a single transaction for fraud.

    Returns the fraud probability, binary flag, decision threshold,
    SHAP values per feature, and top risk factors.
    """
    try:
        model, model_name, threshold = load_model()
        explainer = load_explainer()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Prepare features
    features = prepare_features(transaction.model_dump())
    features_2d = features.reshape(1, -1)

    # Predict
    fraud_prob = float(model.predict_proba(features_2d)[0, 1])
    is_fraud = fraud_prob >= threshold

    # Rule-based overrides (Hybrid Risk System guardrails)
    rule_triggered = False
    rule_reasons = []

    # Rule 1: Escalation for high-amount transactions with moderate ML suspicion
    if transaction.Amount > 300.0 and fraud_prob >= 0.10:
        rule_triggered = True
        rule_reasons.append("High Amount (> $300.00) with elevated ML risk (> 10.0%)")
        is_fraud = True

    # Rule 2 & 3: Behavioral Velocity Checks (Disabled for now)
    # v_triggered, v_reasons = check_velocity_and_record(transaction.card_id, transaction.Amount)
    # if v_triggered:
    #     rule_triggered = True
    #     rule_reasons.extend(v_reasons)
    #     is_fraud = True

    # SHAP explanation
    shap_response = shap_values_to_api_response(
        explainer, features, feature_names=ENGINEERED_FEATURES
    )

    return PredictionResponse(
        fraud_probability=round(fraud_prob, 6),
        is_fraud=is_fraud,
        threshold=threshold,
        shap_values=shap_response["shap_values"],
        top_risk_factors=shap_response["top_risk_factors"],
        rule_triggered=rule_triggered,
        rule_reasons=rule_reasons,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Score multiple transactions in a single request.

    Returns individual predictions for each transaction,
    along with a summary count of flagged fraud.
    """
    try:
        model, model_name, threshold = load_model()
        explainer = load_explainer()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    predictions = []
    fraud_count = 0

    for transaction in request.transactions:
        features = prepare_features(transaction.model_dump())
        features_2d = features.reshape(1, -1)

        fraud_prob = float(model.predict_proba(features_2d)[0, 1])
        is_fraud = fraud_prob >= threshold

        rule_triggered = False
        rule_reasons = []

        if transaction.Amount > 300.0 and fraud_prob >= 0.10:
            rule_triggered = True
            rule_reasons.append("High Amount (> $300.00) with elevated ML risk (> 10.0%)")
            is_fraud = True

        # v_triggered, v_reasons = check_velocity_and_record(transaction.card_id, transaction.Amount)
        # if v_triggered:
        #     rule_triggered = True
        #     rule_reasons.extend(v_reasons)
        #     is_fraud = True

        if is_fraud:
            fraud_count += 1

        shap_response = shap_values_to_api_response(
            explainer, features, feature_names=ENGINEERED_FEATURES
        )

        predictions.append(
            PredictionResponse(
                fraud_probability=round(fraud_prob, 6),
                is_fraud=is_fraud,
                threshold=threshold,
                shap_values=shap_response["shap_values"],
                top_risk_factors=shap_response["top_risk_factors"],
                rule_triggered=rule_triggered,
                rule_reasons=rule_reasons,
            )
        )

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        fraud_count=fraud_count,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    try:
        model, model_name, threshold = load_model()
        status = "healthy"
        model_type = type(model).__name__
    except Exception:
        status = "unhealthy — model not loaded"
        model_name = "none"
        model_type = "none"

    return HealthResponse(
        status=status,
        model_version=model_name,
        model_type=model_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get metadata about the currently loaded model."""
    try:
        model, model_name, threshold = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ModelInfoResponse(
        model_type=type(model).__name__,
        model_name=model_name,
        threshold=threshold,
        features=ENGINEERED_FEATURES,
        n_features=len(ENGINEERED_FEATURES),
    )


@app.get("/sample-transactions")
async def sample_transactions():
    """
    Return a list of predefined sample transactions from the test set.
    The indices correspond to 1-based spreadsheet row numbers (Row 1 is the header).
    """
    return [
        {"index": 2, "label": "Legitimate Transaction #1 ($50.00) — Row 2"},
        {"index": 3, "label": "Legitimate Transaction #2 ($14.95) — Row 3"},
        {"index": 4, "label": "Legitimate Transaction #3 ($7.70) — Row 4"},
        {"index": 5, "label": "Legitimate Transaction #4 ($6.99) — Row 5"},
        {"index": 1870, "label": "Fraudulent Transaction #1 ($1.18) — Row 1870"},
        {"index": 1884, "label": "Fraudulent Transaction #2 ($2.22) — Row 1884"},
        {"index": 2235, "label": "Fraudulent Transaction #3 ($0.77) — Row 2235"},
        {"index": 2635, "label": "Fraudulent Transaction #4 ($94.82) — Row 2635"},
        {"index": 4135, "label": "Fraudulent Transaction #5 ($8.00) — Row 4135"},
        {"index": 6861, "label": "Fraudulent Transaction #6 ($0.00) — Row 6861"},
    ]


@app.get("/sample-transaction/{row_number}")
async def sample_transaction(row_number: int):
    """
    Retrieve features for a given spreadsheet row number from test.csv.
    Row 1 is the header, so data starts at Row 2.
    Strips 'Class' (ground truth) and engineered columns.
    """
    df = load_test_df()
    if df is None:
        raise HTTPException(
            status_code=503,
            detail="Test dataset not loaded on server. Run preparation pipeline first.",
        )

    if row_number < 2 or row_number > len(df) + 1:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid row number. Must be between 2 and {len(df) + 1} (Row 1 is the header row).",
        )

    # Map spreadsheet row number (1-based, row 1 is header) to pandas 0-based index
    pandas_index = row_number - 2
    row = df.iloc[pandas_index].to_dict()
    
    # Strip ground truth class and pre-engineered columns
    for col in ["Class", "log_amount", "hour_of_day"]:
        row.pop(col, None)

    return row
