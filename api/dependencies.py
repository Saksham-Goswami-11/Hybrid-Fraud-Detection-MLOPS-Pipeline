"""
Dependency injection for the FastAPI application.

Handles model loading, SHAP explainer initialization,
and threshold configuration — cached as singletons on startup.
"""

import json
import pickle
from functools import lru_cache

import numpy as np

from src.config import BEST_MODEL_PATH, BEST_THRESHOLD_PATH, ENGINEERED_FEATURES
from src.explain import get_shap_explainer
from src.utils import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_model():
    """
    Load the trained model from disk (cached singleton).

    Returns:
        Tuple of (model, model_name, threshold).
    """
    logger.info(f"Loading model from {BEST_MODEL_PATH}...")

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {BEST_MODEL_PATH}. "
            "Run the training pipeline first: python -m src.train"
        )

    with open(BEST_MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load threshold
    threshold_data = {"threshold": 0.5, "model_name": "unknown"}
    if BEST_THRESHOLD_PATH.exists():
        with open(BEST_THRESHOLD_PATH, "r") as f:
            threshold_data = json.load(f)

    model_name = threshold_data.get("model_name", type(model).__name__)
    threshold = threshold_data.get("threshold", 0.5)

    logger.info(f"  Model loaded: {model_name} (threshold: {threshold:.4f})")
    return model, model_name, threshold


@lru_cache(maxsize=1)
def load_explainer():
    """
    Initialize the SHAP explainer (cached singleton).

    Returns:
        SHAP explainer instance.
    """
    model, model_name, _ = load_model()
    logger.info("Initializing SHAP explainer...")
    explainer = get_shap_explainer(model)
    return explainer


def prepare_features(transaction_dict: dict) -> np.ndarray:
    """
    Convert a transaction input dict to a feature vector
    matching the ENGINEERED_FEATURES format.

    Applies the same feature engineering as data_prep.py:
    - log_amount = log1p(Amount)
    - hour_of_day = (Time % 86400) / 3600

    Args:
        transaction_dict: Dictionary with raw transaction fields.

    Returns:
        1D numpy array with engineered features.
    """
    features = []
    for feat in ENGINEERED_FEATURES:
        if feat == "log_amount":
            features.append(np.log1p(transaction_dict["Amount"]))
        elif feat == "hour_of_day":
            features.append((transaction_dict["Time"] % 86400) / 3600)
        else:
            features.append(transaction_dict[feat])

    return np.array(features, dtype=np.float64)
