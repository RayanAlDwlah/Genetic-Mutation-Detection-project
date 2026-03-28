"""Evaluation utilities for mutation detection models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute binary classification metrics from probabilities."""

    y_true_arr = np.asarray(y_true).astype(int)
    # Clip probabilities away from 0/1 to prevent log(0) in log_loss and brier_score_loss.
    y_prob_arr = np.clip(np.asarray(y_prob).astype(float), 1e-12, 1.0 - 1e-12)
    y_pred = (y_prob_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true_arr, y_prob_arr)),
        "pr_auc": float(average_precision_score(y_true_arr, y_prob_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred)),
        "brier_loss": float(brier_score_loss(y_true_arr, y_prob_arr)),
        "log_loss": float(log_loss(y_true_arr, y_prob_arr, labels=[0, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "support": int(y_true_arr.shape[0]),
        "pathogenic_count": int((y_true_arr == 1).sum()),
        "benign_count": int((y_true_arr == 0).sum()),
    }


def select_best_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    *,
    min_threshold: float = 0.20,
    max_threshold: float = 0.80,
    steps: int = 121,
) -> tuple[float, pd.DataFrame]:
    """Choose threshold that maximizes F1 on validation set.

    Search range [0.20, 0.80] is intentionally conservative — thresholds
    below 0.20 or above 0.80 produce extreme precision/recall trade-offs
    that are not useful in a balanced clinical classification setting.
    The threshold is always tuned on the validation set, never on the test set.
    """

    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    thresholds = np.linspace(min_threshold, max_threshold, steps)
    rows: list[dict[str, float]] = []

    best_t = 0.5
    best_f1 = -1.0

    for t in thresholds:
        y_pred = (y_prob_arr >= t).astype(int)
        f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
        precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
        recall = float(recall_score(y_true_arr, y_pred, zero_division=0))

        rows.append(
            {
                "threshold": float(t),
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
        )

        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    curve = pd.DataFrame(rows)
    return best_t, curve
