"""
Tests for the data preparation module.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_prep import create_features, split_data, validate_data


class TestValidateData:
    """Tests for data validation."""

    def test_valid_data_passes(self, sample_dataframe):
        """Validation should pass for a well-formed DataFrame."""
        results = validate_data(sample_dataframe)
        assert results["columns_valid"] is True
        assert results["no_nulls"] is True
        assert results["target_valid"] is True
        assert results["amount_valid"] is True

    def test_missing_columns_raises(self, sample_dataframe):
        """Validation should raise ValueError if required columns are missing."""
        df = sample_dataframe.drop(columns=["V1", "V2"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_data(df)

    def test_null_values_raises(self, sample_dataframe):
        """Validation should raise ValueError if null values exist."""
        df = sample_dataframe.copy()
        df.loc[0, "V1"] = None
        with pytest.raises(ValueError, match="null values"):
            validate_data(df)

    def test_invalid_target_raises(self, sample_dataframe):
        """Validation should raise ValueError if target has unexpected values."""
        df = sample_dataframe.copy()
        df.loc[0, "Class"] = 2
        with pytest.raises(ValueError, match="unexpected values"):
            validate_data(df)

    def test_negative_amount_raises(self, sample_dataframe):
        """Validation should raise ValueError if Amount is negative."""
        df = sample_dataframe.copy()
        df.loc[0, "Amount"] = -10.0
        with pytest.raises(ValueError, match="negative values"):
            validate_data(df)

    def test_class_distribution_reported(self, sample_dataframe):
        """Validation should report class distribution stats."""
        results = validate_data(sample_dataframe)
        dist = results["class_distribution"]
        assert dist["total"] == len(sample_dataframe)
        assert dist["fraud"] == 3
        assert dist["legitimate"] == 97
        assert 0 < dist["fraud_rate"] < 1


class TestCreateFeatures:
    """Tests for feature engineering."""

    def test_log_amount_created(self, sample_dataframe):
        """Feature engineering should create log_amount column."""
        df = create_features(sample_dataframe)
        assert "log_amount" in df.columns
        assert np.all(np.isfinite(df["log_amount"]))

    def test_hour_of_day_created(self, sample_dataframe):
        """Feature engineering should create hour_of_day column."""
        df = create_features(sample_dataframe)
        assert "hour_of_day" in df.columns
        assert df["hour_of_day"].min() >= 0
        assert df["hour_of_day"].max() < 24

    def test_log_amount_handles_zero(self, sample_dataframe):
        """log1p should handle Amount=0 correctly."""
        df = sample_dataframe.copy()
        df.loc[0, "Amount"] = 0.0
        df = create_features(df)
        assert df.loc[0, "log_amount"] == 0.0

    def test_original_features_preserved(self, sample_dataframe):
        """Original features should not be modified."""
        original_v1 = sample_dataframe["V1"].copy()
        df = create_features(sample_dataframe)
        pd.testing.assert_series_equal(df["V1"], original_v1)


class TestSplitData:
    """Tests for time-aware train/test split."""

    def test_split_preserves_temporal_order(self, sample_dataframe):
        """Train set should contain earlier transactions than test set."""
        train_df, test_df = split_data(sample_dataframe)
        assert train_df["Time"].max() <= test_df["Time"].min()

    def test_split_ratio(self, sample_dataframe):
        """Split should approximate the configured test size."""
        train_df, test_df = split_data(sample_dataframe)
        total = len(train_df) + len(test_df)
        test_ratio = len(test_df) / total
        assert 0.15 <= test_ratio <= 0.25  # Allow some tolerance

    def test_no_data_loss(self, sample_dataframe):
        """All rows should be present in either train or test."""
        train_df, test_df = split_data(sample_dataframe)
        assert len(train_df) + len(test_df) == len(sample_dataframe)
