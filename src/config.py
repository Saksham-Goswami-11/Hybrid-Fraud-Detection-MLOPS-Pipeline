"""
Central configuration for the Fraud Detection System.

All paths, hyperparameters, thresholds, and feature definitions live here
so they can be imported consistently across modules.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
MONITORING_DIR = PROJECT_ROOT / "monitoring" / "reports"

# Ensure directories exist
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, MONITORING_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
RAW_DATA_FILE = DATA_RAW_DIR / "creditcard.csv"
TRAIN_DATA_FILE = DATA_PROCESSED_DIR / "train.csv"
TEST_DATA_FILE = DATA_PROCESSED_DIR / "test.csv"

# ──────────────────────────────────────────────
# Feature definitions
# ──────────────────────────────────────────────
TARGET_COL = "Class"
TIME_COL = "Time"
AMOUNT_COL = "Amount"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]  # V1 through V28
ALL_FEATURES = PCA_FEATURES + [TIME_COL, AMOUNT_COL]

# Features used for modeling (after preprocessing)
# Time is transformed to hour_of_day; Amount is log-transformed
ENGINEERED_FEATURES = PCA_FEATURES + ["log_amount", "hour_of_day"]

# ──────────────────────────────────────────────
# Data split
# ──────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.2  # 80/20 time-aware split

# ──────────────────────────────────────────────
# Business cost matrix
# ──────────────────────────────────────────────
# Cost of missing a fraudulent transaction (false negative)
FRAUD_MISS_COST = 500.0
# Cost of a false alarm (false positive) — analyst review time
FALSE_ALARM_COST = 10.0

# ──────────────────────────────────────────────
# Model training defaults
# ──────────────────────────────────────────────
OPTUNA_N_TRIALS = 100
OPTUNA_CV_FOLDS = 3
DEFAULT_THRESHOLD = 0.5

# ──────────────────────────────────────────────
# MLflow
# ──────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "fraud-detection"
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")

# ──────────────────────────────────────────────
# API
# ──────────────────────────────────────────────
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
BEST_THRESHOLD_PATH = MODELS_DIR / "best_threshold.json"
API_HOST = "0.0.0.0"
API_PORT = 8000

# ──────────────────────────────────────────────
# Device (PyTorch — Apple Silicon MPS)
# ──────────────────────────────────────────────
import torch

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
