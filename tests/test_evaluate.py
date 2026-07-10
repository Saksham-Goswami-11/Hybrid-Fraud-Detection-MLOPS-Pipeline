"""
Tests for the evaluation module.
"""

import numpy as np
import pytest

from src.evaluate import (
    compute_business_cost,
    compute_metrics,
    find_optimal_threshold,
)


class TestComputeMetrics:
    """Tests for metrics computation."""

    def test_returns_expected_keys(self, sample_predictions):
        """Metrics dict should contain all expected keys."""
        y_true, y_pred, y_prob = sample_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob)

        expected_keys = {
            "pr_auc", "roc_auc", "f1", "f2", "threshold",
            "true_positives", "false_positives",
            "true_negatives", "false_negatives",
            "precision", "recall",
        }
        assert expected_keys.issubset(metrics.keys())

    def test_metrics_in_valid_range(self, sample_predictions):
        """All probability-based metrics should be in [0, 1]."""
        y_true, y_pred, y_prob = sample_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob)

        for key in ["pr_auc", "roc_auc", "f1", "f2", "precision", "recall"]:
            assert 0 <= metrics[key] <= 1, f"{key} = {metrics[key]} out of range"

    def test_confusion_matrix_sums_to_total(self, sample_predictions):
        """TP + FP + TN + FN should equal total samples."""
        y_true, y_pred, y_prob = sample_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob)

        total = (
            metrics["true_positives"]
            + metrics["false_positives"]
            + metrics["true_negatives"]
            + metrics["false_negatives"]
        )
        assert total == len(y_true)

    def test_perfect_predictions(self):
        """Perfect predictions should yield precision=1, recall=1."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.95])

        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 0


class TestBusinessCost:
    """Tests for business cost calculation."""

    def test_no_errors_zero_cost(self):
        """Perfect predictions should have zero cost."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        cost = compute_business_cost(y_true, y_pred)
        assert cost == 0.0

    def test_missed_fraud_cost(self):
        """Missing one fraud should cost FRAUD_MISS_COST."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1])  # One missed fraud
        cost = compute_business_cost(y_true, y_pred, fraud_cost=500, fp_cost=10)
        assert cost == 500.0  # 1 FN × $500

    def test_false_alarm_cost(self):
        """One false alarm should cost FALSE_ALARM_COST."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 0, 1, 1])  # One false positive
        cost = compute_business_cost(y_true, y_pred, fraud_cost=500, fp_cost=10)
        assert cost == 10.0  # 1 FP × $10

    def test_combined_cost(self):
        """Combined errors should sum correctly."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1])  # 1 FP + 1 FN
        cost = compute_business_cost(y_true, y_pred, fraud_cost=500, fp_cost=10)
        assert cost == 510.0  # (1 × $500) + (1 × $10)


class TestFindOptimalThreshold:
    """Tests for optimal threshold search."""

    def test_returns_float_in_range(self, sample_predictions):
        """Optimal threshold should be a float in (0, 1)."""
        y_true, _, y_prob = sample_predictions
        threshold = find_optimal_threshold(y_true, y_prob, metric="f2")
        assert isinstance(threshold, float)
        assert 0 < threshold < 1

    def test_different_metrics_different_thresholds(self, sample_predictions):
        """Different optimization metrics may produce different thresholds."""
        y_true, _, y_prob = sample_predictions
        t_f1 = find_optimal_threshold(y_true, y_prob, metric="f1")
        t_f2 = find_optimal_threshold(y_true, y_prob, metric="f2")
        # F2 weighs recall more, so threshold is typically lower
        # (We just verify both return valid values; they may coincide on small data)
        assert 0 < t_f1 < 1
        assert 0 < t_f2 < 1
