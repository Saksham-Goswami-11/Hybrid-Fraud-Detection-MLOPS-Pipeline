"""
Simulate data drift for testing the monitoring pipeline.

Injects synthetic distribution shifts into the test data to verify
that the Evidently drift detection system catches them.
"""

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    ENGINEERED_FEATURES,
    TARGET_COL,
    TEST_DATA_FILE,
)
from src.utils import get_logger, set_global_seed

logger = get_logger(__name__)


def simulate_amount_drift(df: pd.DataFrame, multiplier: float = 2.0) -> pd.DataFrame:
    """
    Shift the log_amount distribution by a multiplier.

    Simulates a scenario where transaction amounts increase over time.
    """
    df = df.copy()
    df["log_amount"] = df["log_amount"] * multiplier
    logger.info(f"  Applied amount drift: log_amount × {multiplier}")
    return df


def simulate_feature_noise(
    df: pd.DataFrame,
    features: list[str] = None,
    noise_std: float = 1.0,
) -> pd.DataFrame:
    """
    Add Gaussian noise to selected PCA features.

    Simulates gradual feature distribution shift.
    """
    df = df.copy()
    features = features or [f"V{i}" for i in range(1, 6)]  # V1–V5

    for feat in features:
        if feat in df.columns:
            noise = np.random.normal(0, noise_std, size=len(df))
            df[feat] = df[feat] + noise

    logger.info(f"  Applied noise (std={noise_std}) to features: {features}")
    return df


def simulate_fraud_rate_increase(
    df: pd.DataFrame,
    target_fraud_rate: float = 0.01,
) -> pd.DataFrame:
    """
    Increase the fraud rate by duplicating fraud samples.

    Simulates a scenario where fraud becomes more prevalent.
    """
    df = df.copy()
    current_fraud = df[TARGET_COL].sum()
    current_total = len(df)
    current_rate = current_fraud / current_total

    if current_rate >= target_fraud_rate:
        logger.info(f"  Fraud rate ({current_rate:.4%}) already >= target ({target_fraud_rate:.4%})")
        return df

    # Calculate how many fraud samples to add
    fraud_samples = df[df[TARGET_COL] == 1]
    n_needed = int((target_fraud_rate * current_total) / (1 - target_fraud_rate)) - int(
        current_fraud
    )
    n_needed = max(0, n_needed)

    if n_needed > 0 and len(fraud_samples) > 0:
        # Duplicate fraud samples with slight noise
        extra_fraud = fraud_samples.sample(n=n_needed, replace=True, random_state=42)
        # Add small noise to avoid exact duplicates
        for feat in ENGINEERED_FEATURES:
            if feat in extra_fraud.columns:
                noise = np.random.normal(0, 0.01, size=len(extra_fraud))
                extra_fraud[feat] = extra_fraud[feat] + noise

        df = pd.concat([df, extra_fraud], ignore_index=True)
        new_rate = df[TARGET_COL].sum() / len(df)
        logger.info(
            f"  Increased fraud rate: {current_rate:.4%} → {new_rate:.4%} "
            f"(added {n_needed} synthetic fraud samples)"
        )

    return df


def generate_drifted_dataset(
    apply_amount_drift: bool = True,
    apply_noise: bool = True,
    apply_fraud_increase: bool = True,
    output_path=None,
) -> pd.DataFrame:
    """
    Generate a drifted version of the test dataset.

    Applies multiple drift types to simulate realistic production drift.

    Args:
        apply_amount_drift: Whether to shift Amount distribution.
        apply_noise: Whether to add noise to PCA features.
        apply_fraud_increase: Whether to increase fraud rate.
        output_path: Path to save drifted dataset.

    Returns:
        Drifted DataFrame.
    """
    set_global_seed()
    logger.info("Generating drifted dataset...")

    df = pd.read_csv(TEST_DATA_FILE)

    if apply_amount_drift:
        df = simulate_amount_drift(df)

    if apply_noise:
        df = simulate_feature_noise(df)

    if apply_fraud_increase:
        df = simulate_fraud_rate_increase(df)

    output_path = output_path or DATA_PROCESSED_DIR / "test_drifted.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"  Saved drifted dataset to {output_path}")

    return df


if __name__ == "__main__":
    from monitoring.drift_report import generate_data_drift_report

    # Generate drifted data
    drifted_df = generate_drifted_dataset()

    # Generate drift report comparing reference (train) vs drifted (test)
    reference_df = pd.read_csv(
        DATA_PROCESSED_DIR.parent / "processed" / "train.csv"
        if not (DATA_PROCESSED_DIR / "train.csv").exists()
        else DATA_PROCESSED_DIR / "train.csv"
    )

    generate_data_drift_report(
        reference_df=reference_df,
        current_df=drifted_df,
        output_path=None,  # Uses default path
    )

    logger.info("Drift simulation complete ✓")
