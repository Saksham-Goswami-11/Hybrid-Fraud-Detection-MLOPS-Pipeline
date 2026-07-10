"""
Drift detection and monitoring using Evidently AI.

Generates data drift and model performance reports
comparing reference (training) data against current (production) data.
"""

import pandas as pd
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import (
    DataDriftPreset,
    ClassificationPreset,
    TargetDriftPreset,
)

from src.config import (
    ENGINEERED_FEATURES,
    MONITORING_DIR,
    TARGET_COL,
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
)
from src.utils import get_logger

logger = get_logger(__name__)


def create_column_mapping() -> ColumnMapping:
    """Create Evidently column mapping for our dataset."""
    return ColumnMapping(
        target=TARGET_COL,
        prediction="prediction",
        numerical_features=ENGINEERED_FEATURES,
    )


def generate_data_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path=None,
) -> Report:
    """
    Generate a data drift report comparing reference and current data.

    Args:
        reference_df: Training/reference data.
        current_df: New/current data to check for drift.
        output_path: Path to save the HTML report.

    Returns:
        Evidently Report object.
    """
    logger.info("Generating data drift report...")

    column_mapping = ColumnMapping(
        numerical_features=ENGINEERED_FEATURES,
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df[ENGINEERED_FEATURES],
        current_data=current_df[ENGINEERED_FEATURES],
        column_mapping=column_mapping,
    )

    output_path = output_path or MONITORING_DIR / "data_drift_report.html"
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    logger.info(f"  Saved data drift report to {output_path}")

    return report


def generate_performance_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path=None,
) -> Report:
    """
    Generate a classification performance report.

    Both DataFrames must include the target column and a 'prediction' column.

    Args:
        reference_df: Reference data with target and predictions.
        current_df: Current data with target and predictions.
        output_path: Path to save the HTML report.

    Returns:
        Evidently Report object.
    """
    logger.info("Generating classification performance report...")

    column_mapping = create_column_mapping()

    report = Report(metrics=[ClassificationPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    output_path = output_path or MONITORING_DIR / "performance_report.html"
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    logger.info(f"  Saved performance report to {output_path}")

    return report


def generate_target_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path=None,
) -> Report:
    """
    Generate a target/prediction distribution drift report.

    Args:
        reference_df: Reference data with predictions.
        current_df: Current data with predictions.
        output_path: Path to save the HTML report.

    Returns:
        Evidently Report object.
    """
    logger.info("Generating target drift report...")

    column_mapping = create_column_mapping()

    report = Report(metrics=[TargetDriftPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    output_path = output_path or MONITORING_DIR / "target_drift_report.html"
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    logger.info(f"  Saved target drift report to {output_path}")

    return report


if __name__ == "__main__":
    # Generate drift report using train/test data
    logger.info("Running drift analysis with train (reference) vs test (current)...")

    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)

    generate_data_drift_report(train_df, test_df)
    logger.info("Drift report generation complete ✓")
