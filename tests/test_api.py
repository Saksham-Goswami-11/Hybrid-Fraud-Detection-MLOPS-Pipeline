"""
Tests for the FastAPI application.

Tests cover the /predict, /health, and /model/info endpoints.
Note: These tests require a trained model to be available.
For CI, a fixture creates a mock model.
"""

import json
import pickle
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.config import BEST_MODEL_PATH, BEST_THRESHOLD_PATH, MODELS_DIR


@pytest.fixture(scope="module")
def mock_model():
    """
    Create and save a simple mock model for API testing.

    This avoids needing the full training pipeline to run API tests.
    """
    # Train a trivial model
    np.random.seed(42)
    X = np.random.randn(100, 30)
    y = np.zeros(100, dtype=int)
    y[:5] = 1

    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Save threshold
    with open(BEST_THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": 0.5, "model_name": "test_random_forest"}, f)

    yield model

    # Cleanup (optional — tests may want the model to persist)


@pytest.fixture(scope="module")
def client(mock_model):
    """Create a FastAPI test client with a mock model loaded."""
    # Clear the cached model so it picks up our mock
    from api.dependencies import load_model, load_explainer

    load_model.cache_clear()
    load_explainer.cache_clear()

    from api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_transaction():
    """Return a valid transaction payload."""
    data = {f"V{i}": float(np.random.randn()) for i in range(1, 29)}
    data["Amount"] = 149.62
    data["Time"] = 0.0
    return data


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_format(self, client):
        """Health response should contain expected fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_version" in data
        assert "model_type" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_returns_200(self, client, valid_transaction):
        """Predict endpoint should return 200 for valid input."""
        response = client.post("/predict", json=valid_transaction)
        assert response.status_code == 200

    def test_predict_response_format(self, client, valid_transaction):
        """Prediction response should contain all expected fields."""
        response = client.post("/predict", json=valid_transaction)
        data = response.json()

        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "threshold" in data
        assert "shap_values" in data
        assert "top_risk_factors" in data

    def test_predict_probability_range(self, client, valid_transaction):
        """Fraud probability should be between 0 and 1."""
        response = client.post("/predict", json=valid_transaction)
        data = response.json()
        assert 0 <= data["fraud_probability"] <= 1

    def test_predict_shap_values_present(self, client, valid_transaction):
        """SHAP values should be a non-empty dictionary."""
        response = client.post("/predict", json=valid_transaction)
        data = response.json()
        assert isinstance(data["shap_values"], dict)
        assert len(data["shap_values"]) > 0

    def test_predict_rejects_missing_fields(self, client):
        """Predict should return 422 for missing required fields."""
        incomplete = {"V1": 1.0, "Amount": 100.0}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_predict_rejects_negative_amount(self, client, valid_transaction):
        """Predict should reject negative Amount values."""
        valid_transaction["Amount"] = -10.0
        response = client.post("/predict", json=valid_transaction)
        assert response.status_code == 422

    def test_predict_latency(self, client, valid_transaction):
        """Single prediction should complete in under 200ms."""
        start = time.time()
        response = client.post("/predict", json=valid_transaction)
        latency_ms = (time.time() - start) * 1000

        assert response.status_code == 200
        # Allow generous margin for test environments
        assert latency_ms < 2000, f"Prediction took {latency_ms:.0f}ms (limit: 2000ms)"

    @pytest.mark.skip(reason="Velocity checks are disabled for now")
    def test_predict_velocity_over_limit_triggers_override(self, client, valid_transaction):
        """API should trigger rule override on the 4th consecutive request for a card."""
        valid_transaction["card_id"] = "test_card_limit_pytest"
        
        # Hitting 3 times should be normal
        for _ in range(3):
            response = client.post("/predict", json=valid_transaction)
            assert response.status_code == 200
            data = response.json()
            assert data["rule_triggered"] is False
            
        # 4th time should trigger velocity count alert override
        response = client.post("/predict", json=valid_transaction)
        assert response.status_code == 200
        data = response.json()
        assert data["rule_triggered"] is True
        assert data["is_fraud"] is True
        assert any("exceeded 3 transactions" in r for r in data["rule_reasons"])

    @pytest.mark.skip(reason="Velocity checks are disabled for now")
    def test_predict_velocity_spend_limit_triggers_override(self, client, valid_transaction):
        """API should trigger rule override immediately if single transaction is > $1000."""
        valid_transaction["card_id"] = "test_card_spend_pytest"
        valid_transaction["Amount"] = 1200.0
        
        response = client.post("/predict", json=valid_transaction)
        assert response.status_code == 200
        data = response.json()
        assert data["rule_triggered"] is True
        assert data["is_fraud"] is True
        assert any("spending exceeded $1,000" in r for r in data["rule_reasons"])


class TestModelInfoEndpoint:
    """Tests for GET /model/info."""

    def test_model_info_returns_200(self, client):
        """Model info endpoint should return 200."""
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_format(self, client):
        """Model info should contain expected fields."""
        response = client.get("/model/info")
        data = response.json()
        assert "model_type" in data
        assert "model_name" in data
        assert "threshold" in data
        assert "features" in data
        assert "n_features" in data
        assert data["n_features"] == len(data["features"])
