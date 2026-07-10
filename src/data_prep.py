"""
Data preparation pipeline for the Fraud Detection System.

Handles loading, validation, feature engineering, and time-aware
train/test splitting of the credit card fraud dataset.

Can be run as a module: python -m src.data_prep
"""

import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_FILE,
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    TARGET_COL,
    TIME_COL,
    AMOUNT_COL,
    PCA_FEATURES,
    ALL_FEATURES,
    TEST_SIZE,
)
from src.utils import get_logger, set_global_seed

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────
def load_data(filepath=None) -> pd.DataFrame:
    """
    Load the raw credit card fraud dataset.

    Args:
        filepath: Path to the CSV file. Defaults to RAW_DATA_FILE from config.

    Returns:
        Raw DataFrame with all columns.
    """
    filepath = filepath or RAW_DATA_FILE
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} transactions with {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────
# Data Validation
# ──────────────────────────────────────────────
def validate_data(df: pd.DataFrame) -> dict:
    """
    Run validation checks on the dataset.

    Checks:
        - Required columns are present
        - No null values
        - Target column contains only 0 and 1
        - Amount is non-negative
        - Class distribution is reported

    Args:
        df: DataFrame to validate.

    Returns:
        Dictionary with validation results and class distribution stats.

    Raises:
        ValueError: If any critical validation check fails.
    """
    logger.info("Running data validation checks...")
    results = {}

    # Check required columns
    required_cols = ALL_FEATURES + [TARGET_COL]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    results["columns_valid"] = True
    logger.info("  ✓ All required columns present")

    # Check for null values
    null_counts = df[required_cols].isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        raise ValueError(f"Found {total_nulls} null values:\n{null_counts[null_counts > 0]}")
    results["no_nulls"] = True
    logger.info("  ✓ No null values found")

    # Check target column values
    unique_targets = df[TARGET_COL].unique()
    if not set(unique_targets).issubset({0, 1}):
        raise ValueError(f"Target column contains unexpected values: {unique_targets}")
    results["target_valid"] = True
    logger.info("  ✓ Target column contains only 0 and 1")

    # Check Amount is non-negative
    if (df[AMOUNT_COL] < 0).any():
        raise ValueError("Found negative values in Amount column")
    results["amount_valid"] = True
    logger.info("  ✓ Amount values are non-negative")

    # Class distribution
    class_counts = df[TARGET_COL].value_counts()
    fraud_rate = class_counts.get(1, 0) / len(df)
    results["class_distribution"] = {
        "total": len(df),
        "legitimate": int(class_counts.get(0, 0)),
        "fraud": int(class_counts.get(1, 0)),
        "fraud_rate": fraud_rate,
    }
    logger.info(
        f"  ✓ Class distribution: {class_counts.get(0, 0):,} legitimate, "
        f"{class_counts.get(1, 0):,} fraud ({fraud_rate:.4%})"
    )

    logger.info("All validation checks passed ✓")
    return results


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering transformations.

    Transformations:
        - Log-transform Amount (handles right-skewed distribution)
        - Convert Time to hour_of_day (cyclic pattern extraction)
        - Keep all original PCA features unchanged

    Args:
        df: DataFrame with raw features.

    Returns:
        DataFrame with engineered features added.
    """
    logger.info("Engineering features...")
    df = df.copy()

    # Log-transform Amount (add 1 to handle zero amounts)
    df["log_amount"] = np.log1p(df[AMOUNT_COL])
    logger.info("  ✓ Created log_amount (log1p transform)")

    # Convert Time (seconds from first transaction) to hour of day
    # Time wraps around every 24 hours (86400 seconds)
    df["hour_of_day"] = (df[TIME_COL] % 86400) / 3600
    logger.info("  ✓ Created hour_of_day (cyclic time feature)")

    return df


# ──────────────────────────────────────────────
# Train/Test Split
# ──────────────────────────────────────────────
def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a time-aware train/test split.

    The dataset is sorted by Time and split at the (1 - TEST_SIZE) quantile.
    This prevents temporal leakage: the model only trains on past data
    and is tested on future data.

    Args:
        df: Full DataFrame sorted by Time.

    Returns:
        Tuple of (train_df, test_df).
    """
    logger.info("Performing time-aware train/test split...")

    # Sort by Time to ensure temporal ordering
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    # Split at the threshold index
    split_idx = int(len(df) * (1 - TEST_SIZE))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Report split statistics
    train_fraud = train_df[TARGET_COL].sum()
    test_fraud = test_df[TARGET_COL].sum()
    logger.info(
        f"  Train: {len(train_df):,} transactions "
        f"({train_fraud:.0f} fraud, {train_fraud / len(train_df):.4%})"
    )
    logger.info(
        f"  Test:  {len(test_df):,} transactions "
        f"({test_fraud:.0f} fraud, {test_fraud / len(test_df):.4%})"
    )

    return train_df, test_df


# ──────────────────────────────────────────────
# Save Processed Data
# ──────────────────────────────────────────────
def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Save train and test DataFrames to CSV files.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
    """
    TRAIN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_DATA_FILE, index=False)
    logger.info(f"  Saved training data to {TRAIN_DATA_FILE}")

    test_df.to_csv(TEST_DATA_FILE, index=False)
    logger.info(f"  Saved test data to {TEST_DATA_FILE}")


# ──────────────────────────────────────────────
# Full Pipeline (CLI entry point)
# ──────────────────────────────────────────────
def run_pipeline() -> None:
    """
    Run the complete data preparation pipeline:
    load → validate → engineer features → split → save.
    """
    set_global_seed()

    # Load
    df = load_data()

    # Validate
    validate_data(df)

    # Feature engineering
    df = create_features(df)

    # Split
    train_df, test_df = split_data(df)

    # Save
    save_splits(train_df, test_df)

    logger.info("Data preparation pipeline complete ✓")


if __name__ == "__main__":
    run_pipeline()
