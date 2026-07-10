"""
Shared test fixtures for the Fraud Detection System.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_transaction_dict():
    """Return a sample transaction as a dictionary (raw input format)."""
    data = {f"V{i}": float(np.random.randn()) for i in range(1, 29)}
    data["Amount"] = 149.62
    data["Time"] = 0.0
    return data


@pytest.fixture
def sample_dataframe():
    """Return a small sample DataFrame mimicking the raw dataset."""
    np.random.seed(42)
    n = 100
    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data["Amount"] = np.abs(np.random.randn(n)) * 100
    data["Time"] = np.sort(np.random.uniform(0, 172800, n))
    data["Class"] = np.zeros(n, dtype=int)
    data["Class"][:3] = 1  # 3 fraud cases

    return pd.DataFrame(data)


@pytest.fixture
def sample_features_and_labels():
    """Return sample features (X) and labels (y) for model testing."""
    np.random.seed(42)
    n = 200
    n_features = 30  # V1–V28 + log_amount + hour_of_day

    X = np.random.randn(n, n_features)
    y = np.zeros(n, dtype=int)
    y[:10] = 1  # 10 fraud cases

    return X, y


@pytest.fixture
def sample_predictions():
    """Return sample ground truth, predictions, and probabilities."""
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.9, 0.4, 0.15, 0.05, 0.85])

    return y_true, y_pred, y_prob
