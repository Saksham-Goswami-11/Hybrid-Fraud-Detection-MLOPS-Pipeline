"""
FastAPI application for the Fraud Detection System.

Endpoints:
    POST /predict       — Score a single transaction
    POST /predict/batch — Score multiple transactions
    GET  /health        — Health check
    GET  /model/info    — Model metadata
"""

import time
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import load_explainer, load_model, prepare_features
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
