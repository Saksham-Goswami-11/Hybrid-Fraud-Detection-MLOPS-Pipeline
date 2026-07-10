"""
SHAP explainability module for the Fraud Detection System.

Generates global and local explanations for model predictions,
and provides JSON-serializable output for the API.
"""

import matplotlib.pyplot as plt
import numpy as np
import shap

from src.config import ENGINEERED_FEATURES, PLOTS_DIR
from src.utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Explainer Initialization
# ──────────────────────────────────────────────
def get_shap_explainer(model, X_background: np.ndarray = None):
    """
    Create the appropriate SHAP explainer for the given model type.

    Uses TreeExplainer for tree-based models (fast, exact),
    falls back to KernelExplainer for others.

    Args:
        model: Trained model.
        X_background: Background dataset for KernelExplainer (sample of training data).

    Returns:
        SHAP Explainer instance.
    """
    model_type = type(model).__name__
    logger.info(f"Creating SHAP explainer for {model_type}...")

    # Tree-based models → TreeExplainer
    tree_types = (
        "XGBClassifier",
        "LGBMClassifier",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
    )
    if model_type in tree_types:
        explainer = shap.TreeExplainer(model)
        logger.info("  Using TreeExplainer (exact, fast)")
        return explainer

    # Linear models → LinearExplainer
    if model_type == "LogisticRegression":
        if X_background is not None:
            explainer = shap.LinearExplainer(model, X_background)
            logger.info("  Using LinearExplainer")
            return explainer

    # Fallback → KernelExplainer (slow but universal)
    if X_background is not None:
        # Use a small sample for KernelExplainer (it's slow)
        if len(X_background) > 100:
            idx = np.random.choice(len(X_background), 100, replace=False)
            X_background = X_background[idx]

        def predict_fn(x):
            return model.predict_proba(x)[:, 1]

        explainer = shap.KernelExplainer(predict_fn, X_background)
        logger.info("  Using KernelExplainer (slow, universal)")
        return explainer

    raise ValueError(
        f"Cannot create explainer for {model_type} without background data. "
        "Pass X_background parameter."
    )


# ──────────────────────────────────────────────
# Global Explanations
# ──────────────────────────────────────────────
def compute_global_importance(
    explainer,
    X: np.ndarray,
    feature_names: list[str] = None,
    max_samples: int = 1000,
    save_plots: bool = True,
) -> dict:
    """
    Compute global feature importance using SHAP values.

    Args:
        explainer: SHAP explainer instance.
        X: Feature matrix (uses a sample for efficiency).
        feature_names: List of feature names.
        max_samples: Maximum samples to compute SHAP values for.
        save_plots: Whether to save plots to disk.

    Returns:
        Dictionary with feature importance rankings.
    """
    feature_names = feature_names or ENGINEERED_FEATURES
    logger.info(f"Computing global SHAP values on {min(len(X), max_samples)} samples...")

    # Sample if dataset is large
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)

    # Handle multi-output (e.g., tree models return [class_0, class_1])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take fraud class

    # Compute mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_names, mean_abs_shap.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    logger.info("  Top 10 features by SHAP importance:")
    for i, (feat, val) in enumerate(list(importance.items())[:10]):
        logger.info(f"    {i+1}. {feat}: {val:.4f}")

    # Generate plots
    if save_plots:
        # Summary (beeswarm) plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved SHAP summary plot to {PLOTS_DIR / 'shap_summary.png'}")

        # Bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved SHAP bar plot to {PLOTS_DIR / 'shap_bar.png'}")

    return importance


# ──────────────────────────────────────────────
# Local (Per-Instance) Explanations
# ──────────────────────────────────────────────
def compute_local_explanation(
    explainer,
    instance: np.ndarray,
    feature_names: list[str] = None,
) -> dict:
    """
    Compute SHAP explanation for a single transaction.

    Args:
        explainer: SHAP explainer instance.
        instance: Single feature vector (1D or 2D with shape (1, n_features)).
        feature_names: List of feature names.

    Returns:
        Dictionary with:
            - shap_values: {feature_name: SHAP contribution}
            - top_risk_factors: top 5 features pushing toward fraud
            - base_value: expected value (average model output)
    """
    feature_names = feature_names or ENGINEERED_FEATURES

    # Ensure 2D input
    if instance.ndim == 1:
        instance = instance.reshape(1, -1)

    # Compute SHAP values
    shap_values = explainer.shap_values(instance)

    # Handle multi-output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Fraud class

    shap_values = shap_values.flatten()

    # Build explanation dict
    shap_dict = dict(zip(feature_names, shap_values.tolist()))

    # Top risk factors (features pushing toward fraud, i.e., positive SHAP)
    sorted_features = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
    top_risk = [f"{name} ({val:+.4f})" for name, val in sorted_features[:5]]

    # Get base value
    if hasattr(explainer, "expected_value"):
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        base_value = float(base_value)
    else:
        base_value = None

    return {
        "shap_values": shap_dict,
        "top_risk_factors": top_risk,
        "base_value": base_value,
    }


# ──────────────────────────────────────────────
# API-Ready Serialization
# ──────────────────────────────────────────────
def shap_values_to_api_response(
    explainer,
    instance: np.ndarray,
    feature_names: list[str] = None,
) -> dict:
    """
    Generate SHAP explanation formatted for the API response.

    Args:
        explainer: SHAP explainer instance.
        instance: Single feature vector.
        feature_names: List of feature names.

    Returns:
        Dictionary ready to be included in PredictionResponse.
    """
    explanation = compute_local_explanation(explainer, instance, feature_names)

    return {
        "shap_values": explanation["shap_values"],
        "top_risk_factors": explanation["top_risk_factors"],
    }
