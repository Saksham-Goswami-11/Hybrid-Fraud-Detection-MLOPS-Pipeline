"""
Training orchestrator for the Fraud Detection System.

Trains all model types (supervised and unsupervised), handles
imbalance strategies, and logs experiments to MLflow.

Can be run as a module: python -m src.train
"""

import os

# Fix macOS OpenMP crash — must be set BEFORE importing XGBoost/LightGBM/sklearn
os.environ["OMP_NUM_THREADS"] = "1"

import json
import pickle

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from src.config import (
    BEST_MODEL_PATH,
    BEST_THRESHOLD_PATH,
    DEVICE,
    ENGINEERED_FEATURES,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_COL,
    TEST_DATA_FILE,
    TRAIN_DATA_FILE,
)
from src.evaluate import (
    find_optimal_threshold,
    full_evaluation,
    log_to_mlflow,
)
from src.utils import get_logger, set_global_seed

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Imbalance Handling
# ──────────────────────────────────────────────
def apply_resampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a resampling strategy to handle class imbalance.

    Args:
        X_train: Training features.
        y_train: Training labels.
        strategy: One of 'none', 'smote', 'adasyn', 'undersample'.

    Returns:
        Resampled (X_train, y_train).
    """
    if strategy == "none":
        logger.info("  No resampling applied")
        return X_train, y_train
    elif strategy == "smote":
        sampler = SMOTE(random_state=RANDOM_SEED)
    elif strategy == "adasyn":
        sampler = ADASYN(random_state=RANDOM_SEED)
    elif strategy == "undersample":
        sampler = RandomUnderSampler(random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown resampling strategy: {strategy}")

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    logger.info(f"  Resampling ({strategy}): {len(X_train):,} → {len(X_resampled):,} samples")
    return X_resampled, y_resampled


# ──────────────────────────────────────────────
# Supervised Models
# ──────────────────────────────────────────────
def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: str = "balanced",
    **kwargs,
) -> LogisticRegression:
    """Train a Logistic Regression model with class weighting."""
    logger.info("Training Logistic Regression...")
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=RANDOM_SEED,
        solver="lbfgs",
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **kwargs,
) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    logger.info("Training Random Forest...")
    defaults = {
        "n_estimators": 200,
        "max_depth": 15,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": 1,
    }
    defaults.update(kwargs)
    model = RandomForestClassifier(**defaults)
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **kwargs,
) -> XGBClassifier:
    """Train an XGBoost classifier with scale_pos_weight for imbalance."""
    logger.info("Training XGBoost...")
    # Calculate scale_pos_weight from class ratio
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    defaults = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "n_jobs": 1,
    }
    defaults.update(kwargs)
    model = XGBClassifier(**defaults)
    model.fit(X_train, y_train)
    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **kwargs,
):
    """Train a LightGBM classifier."""
    import lightgbm as lgb

    logger.info("Training LightGBM...")

    defaults = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "class_weight": "balanced",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "metric": "average_precision",
        "verbosity": -1,
        "n_jobs": 1,
    }
    defaults.update(kwargs)
    model = lgb.LGBMClassifier(**defaults)
    model.fit(X_train, y_train)
    return model


# ──────────────────────────────────────────────
# Unsupervised Models
# ──────────────────────────────────────────────
def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.002,
    **kwargs,
) -> IsolationForest:
    """
    Train an Isolation Forest for anomaly detection.

    Predictions are mapped: -1 (outlier/fraud) → 1, 1 (inlier) → 0
    to match the binary classification convention.
    """
    logger.info("Training Isolation Forest...")
    defaults = {
        "n_estimators": 200,
        "contamination": contamination,
        "random_state": RANDOM_SEED,
        "n_jobs": 1,
    }
    defaults.update(kwargs)
    model = IsolationForest(**defaults)
    model.fit(X_train)
    return model


def predict_isolation_forest(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Get anomaly scores from Isolation Forest.

    Returns probabilities by normalizing the decision function
    scores to [0, 1] range (higher = more anomalous = more likely fraud).
    """
    # decision_function: lower = more anomalous
    scores = model.decision_function(X)
    # Invert and normalize to [0, 1]
    probs = (scores.max() - scores) / (scores.max() - scores.min() + 1e-10)
    return probs


# ──────────────────────────────────────────────
# Autoencoder (PyTorch)
# ──────────────────────────────────────────────
class FraudAutoencoder(nn.Module):
    """
    Autoencoder for fraud detection via reconstruction error.

    Architecture:
        Encoder: input_dim → 20 → 14 → 7
        Decoder: 7 → 14 → 20 → input_dim

    Trained only on legitimate transactions. Fraud transactions
    should have higher reconstruction error.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Linear(20, 14),
            nn.ReLU(),
            nn.BatchNorm1d(14),
            nn.Linear(14, 7),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(7, 14),
            nn.ReLU(),
            nn.BatchNorm1d(14),
            nn.Linear(14, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Linear(20, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(
    X_train: np.ndarray,
    y_train: np.ndarray = None,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
) -> tuple[FraudAutoencoder, StandardScaler]:
    """
    Train an autoencoder on legitimate transactions only.

    Args:
        X_train: Training features.
        y_train: Training labels (used to filter legitimate transactions).
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.

    Returns:
        Tuple of (trained model, fitted scaler).
    """
    logger.info(f"Training Autoencoder on {DEVICE}...")

    # Scale features
    scaler = StandardScaler()

    # Train only on legitimate transactions
    if y_train is not None:
        X_legit = X_train[y_train == 0]
    else:
        X_legit = X_train

    X_scaled = scaler.fit_transform(X_legit)
    logger.info(f"  Training on {len(X_scaled):,} legitimate transactions")

    # Create PyTorch dataset
    tensor_data = torch.FloatTensor(X_scaled).to(DEVICE)
    dataset = TensorDataset(tensor_data, tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    input_dim = X_scaled.shape[1]
    model = FraudAutoencoder(input_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, _ in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.6f}")

    model.eval()
    return model, scaler


def predict_autoencoder(
    model: FraudAutoencoder,
    scaler: StandardScaler,
    X: np.ndarray,
) -> np.ndarray:
    """
    Get anomaly scores from autoencoder (reconstruction error).

    Higher reconstruction error = more likely to be fraud.
    Scores are normalized to [0, 1] range.
    """
    model.eval()
    X_scaled = scaler.transform(X)
    tensor_x = torch.FloatTensor(X_scaled).to(DEVICE)

    with torch.no_grad():
        reconstructed = model(tensor_x)

    # Reconstruction error per sample (MSE)
    errors = ((tensor_x - reconstructed) ** 2).mean(dim=1).cpu().numpy()

    # Normalize to [0, 1]
    if errors.max() - errors.min() > 0:
        probs = (errors - errors.min()) / (errors.max() - errors.min())
    else:
        probs = np.zeros_like(errors)

    return probs


# ──────────────────────────────────────────────
# Experiment Runner
# ──────────────────────────────────────────────
def run_experiment(
    model_name: str,
    model,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    params: dict,
    tags: dict = None,
) -> dict:
    """
    Run evaluation and log to MLflow for a single model.

    Args:
        model_name: Human-readable name for the model.
        model: Trained model object.
        y_true: Ground truth test labels.
        y_prob: Predicted probabilities on test set.
        params: Hyperparameters to log.
        tags: Optional tags.

    Returns:
        Dictionary with all metrics.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"{'='*60}")

    # Full evaluation with plots
    metrics = full_evaluation(y_true, y_prob, model_name=model_name)

    # Log to MLflow
    run_tags = {"model_type": model_name}
    if tags:
        run_tags.update(tags)

    log_to_mlflow(
        metrics=metrics,
        params=params,
        model=model if not isinstance(model, FraudAutoencoder) else None,
        model_name=model_name.lower().replace(" ", "_"),
        tags=run_tags,
    )

    return metrics


# ──────────────────────────────────────────────
# Save Best Model
# ──────────────────────────────────────────────
def save_best_model(model, threshold: float, model_name: str) -> None:
    """
    Save the best model and its optimal threshold.

    Args:
        model: Trained model to save.
        threshold: Optimal decision threshold.
        model_name: Name for logging.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"  Saved best model ({model_name}) to {BEST_MODEL_PATH}")

    # Save threshold
    threshold_data = {"threshold": threshold, "model_name": model_name}
    with open(BEST_THRESHOLD_PATH, "w") as f:
        json.dump(threshold_data, f, indent=2)
    logger.info(f"  Saved threshold ({threshold:.4f}) to {BEST_THRESHOLD_PATH}")


# ──────────────────────────────────────────────
# Main Training Pipeline
# ──────────────────────────────────────────────
def run_pipeline() -> None:
    """
    Run the full training pipeline:
    1. Load processed data
    2. Train baseline (Logistic Regression)
    3. Train all models
    4. Compare and select best
    5. Save best model
    """
    set_global_seed()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Load data
    logger.info("Loading processed data...")
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)

    X_train = train_df[ENGINEERED_FEATURES].values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[ENGINEERED_FEATURES].values
    y_test = test_df[TARGET_COL].values

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    all_results = {}

    # ── 1. Logistic Regression Baseline ──
    lr_model = train_logistic_regression(X_train, y_train)
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
    all_results["Logistic Regression"] = run_experiment(
        "Logistic Regression",
        lr_model,
        y_test,
        y_prob_lr,
        params={"model": "logistic_regression", "class_weight": "balanced"},
    )

    # ── 2. Random Forest ──
    rf_model = train_random_forest(X_train, y_train)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    all_results["Random Forest"] = run_experiment(
        "Random Forest",
        rf_model,
        y_test,
        y_prob_rf,
        params={"model": "random_forest", "n_estimators": 200, "max_depth": 15},
    )

    # ── 3. XGBoost ──
    xgb_model = train_xgboost(X_train, y_train)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    all_results["XGBoost"] = run_experiment(
        "XGBoost",
        xgb_model,
        y_test,
        y_prob_xgb,
        params={"model": "xgboost", "n_estimators": 300, "max_depth": 6},
    )

    # ── 4. LightGBM ──
    lgbm_model = train_lightgbm(X_train, y_train)
    y_prob_lgbm = lgbm_model.predict_proba(X_test)[:, 1]
    all_results["LightGBM"] = run_experiment(
        "LightGBM",
        lgbm_model,
        y_test,
        y_prob_lgbm,
        params={"model": "lightgbm", "n_estimators": 300, "max_depth": 6},
    )

    # ── 5. Isolation Forest ──
    iso_model = train_isolation_forest(X_train)
    y_prob_iso = predict_isolation_forest(iso_model, X_test)
    all_results["Isolation Forest"] = run_experiment(
        "Isolation Forest",
        iso_model,
        y_test,
        y_prob_iso,
        params={"model": "isolation_forest", "n_estimators": 200},
    )

    # ── 6. Autoencoder ──
    ae_model, ae_scaler = train_autoencoder(X_train, y_train)
    y_prob_ae = predict_autoencoder(ae_model, ae_scaler, X_test)
    all_results["Autoencoder"] = run_experiment(
        "Autoencoder",
        ae_model,
        y_test,
        y_prob_ae,
        params={"model": "autoencoder", "epochs": 50, "batch_size": 256},
    )

    # ── Compare and select best ──
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON LEADERBOARD")
    logger.info("=" * 60)

    leaderboard = []
    for name, metrics in all_results.items():
        leaderboard.append(
            {
                "Model": name,
                "PR-AUC": metrics["pr_auc"],
                "ROC-AUC": metrics["roc_auc"],
                "F2": metrics["f2"],
                "Recall": metrics["recall"],
                "Precision": metrics["precision"],
                "Cost ($)": metrics["business_cost"],
            }
        )

    leaderboard_df = pd.DataFrame(leaderboard).sort_values("PR-AUC", ascending=False)
    logger.info(f"\n{leaderboard_df.to_string(index=False)}")

    # Select best model by PR-AUC
    best_name = leaderboard_df.iloc[0]["Model"]
    best_pr_auc = leaderboard_df.iloc[0]["PR-AUC"]
    logger.info(f"\n🏆 Best model: {best_name} (PR-AUC: {best_pr_auc:.4f})")

    # Save the best supervised model
    model_map = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "LightGBM": lgbm_model,
    }

    if best_name in model_map:
        prob_map = {
            "Logistic Regression": y_prob_lr,
            "Random Forest": y_prob_rf,
            "XGBoost": y_prob_xgb,
            "LightGBM": y_prob_lgbm,
        }
        best_threshold = find_optimal_threshold(y_test, prob_map[best_name])
        save_best_model(model_map[best_name], best_threshold, best_name)
    else:
        logger.info(f"  Best model ({best_name}) is unsupervised — saving best supervised instead")
        # Fall back to best supervised model
        supervised_df = leaderboard_df[leaderboard_df["Model"].isin(model_map.keys())]
        if not supervised_df.empty:
            fallback_name = supervised_df.iloc[0]["Model"]
            prob_map = {
                "Logistic Regression": y_prob_lr,
                "Random Forest": y_prob_rf,
                "XGBoost": y_prob_xgb,
                "LightGBM": y_prob_lgbm,
            }
            best_threshold = find_optimal_threshold(y_test, prob_map[fallback_name])
            save_best_model(model_map[fallback_name], best_threshold, fallback_name)

    # Save leaderboard
    leaderboard_path = MODELS_DIR / "leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)
    logger.info(f"  Saved leaderboard to {leaderboard_path}")

    logger.info("\nTraining pipeline complete ✓")


if __name__ == "__main__":
    run_pipeline()
