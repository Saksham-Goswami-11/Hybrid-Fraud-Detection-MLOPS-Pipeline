"""
Evaluation harness for the Fraud Detection System.

Provides metrics computation, business cost analysis, visualization,
and MLflow logging — reused consistently across all model experiments.

Can be run as a standalone evaluation: python -m src.evaluate
"""

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.config import (
    DEFAULT_THRESHOLD,
    FALSE_ALARM_COST,
    FRAUD_MISS_COST,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PLOTS_DIR,
)
from src.utils import format_currency, format_percentage, get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Metrics Computation
# ──────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels (at the given threshold).
        y_prob: Predicted probabilities for the positive (fraud) class.
        threshold: Decision threshold used for y_pred.

    Returns:
        Dictionary of metric names to values.
    """
    metrics = {
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "threshold": threshold,
    }

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update(
        {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        }
    )

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f2",
) -> float:
    """
    Find the optimal decision threshold by sweeping thresholds
    and maximizing the chosen metric.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        metric: Metric to optimize ('f2', 'f1', or 'cost').

    Returns:
        Optimal threshold value.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_score = -np.inf
    best_threshold = DEFAULT_THRESHOLD

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        if metric == "f2":
            score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        elif metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "cost":
            # Minimize cost = maximize negative cost
            cost = compute_business_cost(y_true, y_pred)
            score = -cost
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = t

    logger.info(f"Optimal threshold ({metric}): {best_threshold:.4f} (score: {best_score:.4f})")
    return best_threshold


# ──────────────────────────────────────────────
# Business Cost Analysis
# ──────────────────────────────────────────────
def compute_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fraud_cost: float = FRAUD_MISS_COST,
    fp_cost: float = FALSE_ALARM_COST,
) -> float:
    """
    Compute expected business cost of predictions.

    Cost = (missed frauds × fraud_cost) + (false alarms × fp_cost)

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        fraud_cost: Cost per missed fraud (false negative).
        fp_cost: Cost per false alarm (false positive).

    Returns:
        Total business cost.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fn * fraud_cost) + (fp * fp_cost)
    return total_cost


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────
def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path=None,
) -> plt.Figure:
    """
    Plot Precision-Recall curve with AUC annotation.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        model_name: Name for the legend.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f"{model_name} (PR-AUC = {pr_auc:.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved PR curve to {save_path}")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path=None,
) -> plt.Figure:
    """
    Plot annotated confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        model_name: Title annotation.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved confusion matrix to {save_path}")

    return fig


def plot_cost_vs_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path=None,
) -> plt.Figure:
    """
    Plot business cost as a function of decision threshold.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        model_name: Title annotation.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    costs = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        costs.append(compute_business_cost(y_true, y_pred))

    min_idx = np.argmin(costs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, costs, linewidth=2)
    ax.axvline(
        thresholds[min_idx],
        color="red",
        linestyle="--",
        label=f"Optimal: {thresholds[min_idx]:.3f} (cost: {format_currency(costs[min_idx])})",
    )
    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Total Business Cost ($)", fontsize=12)
    ax.set_title(f"Cost vs. Threshold — {model_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved cost vs threshold plot to {save_path}")

    return fig


# ──────────────────────────────────────────────
# MLflow Logging
# ──────────────────────────────────────────────
def log_to_mlflow(
    metrics: dict,
    params: dict,
    model=None,
    model_name: str = "model",
    artifacts: dict = None,
    tags: dict = None,
) -> str:
    """
    Log an experiment run to MLflow.

    Args:
        metrics: Metric name → value dict.
        params: Hyperparameter name → value dict.
        model: Optional sklearn/xgboost model to log.
        model_name: Name for the logged model artifact.
        artifacts: Optional dict of {name: filepath} to log as artifacts.
        tags: Optional tags for the run.

    Returns:
        MLflow run ID.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        # Log model
        if model is not None:
            mlflow.sklearn.log_model(model, model_name)

        # Log extra artifacts
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(str(path))

        # Log tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        run_id = run.info.run_id
        logger.info(f"  Logged to MLflow: run_id={run_id}")

    return run_id


# ──────────────────────────────────────────────
# Full Evaluation Report
# ──────────────────────────────────────────────
def full_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    threshold: float = None,
    save_plots: bool = True,
) -> dict:
    """
    Run the complete evaluation suite: metrics, plots, and cost analysis.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        model_name: Name of the model for titles and filenames.
        threshold: Decision threshold. If None, finds optimal via F2.
        save_plots: Whether to save plots to PLOTS_DIR.

    Returns:
        Dictionary with all metrics and the chosen threshold.
    """
    logger.info(f"Running full evaluation for {model_name}...")

    # Find optimal threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_prob, metric="f2")

    # Apply threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, threshold)
    metrics["business_cost"] = compute_business_cost(y_true, y_pred)

    # Log results
    logger.info(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
    logger.info(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    logger.info(f"  F2-Score:   {metrics['f2']:.4f}")
    logger.info(f"  Precision:  {format_percentage(metrics['precision'])}")
    logger.info(f"  Recall:     {format_percentage(metrics['recall'])}")
    logger.info(f"  Threshold:  {threshold:.4f}")
    logger.info(f"  Cost:       {format_currency(metrics['business_cost'])}")
    logger.info(
        f"  TP={metrics['true_positives']} FP={metrics['false_positives']} "
        f"FN={metrics['false_negatives']} TN={metrics['true_negatives']}"
    )

    # Generate plots
    if save_plots:
        safe_name = model_name.lower().replace(" ", "_")
        plot_pr_curve(
            y_true,
            y_prob,
            model_name,
            save_path=PLOTS_DIR / f"pr_curve_{safe_name}.png",
        )
        plot_confusion_matrix(
            y_true,
            y_pred,
            model_name,
            save_path=PLOTS_DIR / f"confusion_matrix_{safe_name}.png",
        )
        plot_cost_vs_threshold(
            y_true,
            y_prob,
            model_name,
            save_path=PLOTS_DIR / f"cost_threshold_{safe_name}.png",
        )

    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Print and return sklearn classification report."""
    report = classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"], digits=4)
    logger.info(f"\n{report}")
    return report
