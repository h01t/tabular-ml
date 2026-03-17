"""Model evaluation utilities — metrics computation and visualization."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute all relevant classification metrics.

    Args:
        y_true: True binary labels.
        y_pred_proba: Predicted probabilities for the positive class.
        threshold: Classification threshold.

    Returns:
        Dictionary of metric name → value.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    return {
        "pr_auc": average_precision_score(y_true, y_pred_proba),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "threshold": threshold,
    }


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Find the threshold that maximizes F1 score on the precision-recall curve.

    Args:
        y_true: True binary labels.
        y_pred_proba: Predicted probabilities for the positive class.

    Returns:
        Optimal threshold value.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    # F1 = 2 * (precision * recall) / (precision + recall)
    # Avoid division by zero
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0,
    )
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot the precision-recall curve with PR-AUC annotation.

    Args:
        y_true: True binary labels.
        y_pred_proba: Predicted probabilities.
        model_name: Name for the plot title.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=100)

    plt.close(fig)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        model_name: Name for the plot title.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legitimate", "Fraud"])
    ax.set_yticklabels(["Legitimate", "Fraud"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)

    # Annotate cells
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color=color,
                fontsize=14,
            )

    fig.colorbar(im)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=100)

    plt.close(fig)
    return fig


def format_results_table(results: list[dict[str, Any]]) -> str:
    """Format a list of model results into a readable markdown table.

    Args:
        results: List of dicts with keys: model_name, pr_auc, roc_auc, f1, precision, recall.

    Returns:
        Markdown-formatted table string.
    """
    header = "| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |"
    separator = "|---|---|---|---|---|---|"
    rows = []
    for r in sorted(results, key=lambda x: x["pr_auc"], reverse=True):
        rows.append(
            f"| {r['model_name']} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} "
            f"| {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |"
        )
    return "\n".join([header, separator] + rows)
