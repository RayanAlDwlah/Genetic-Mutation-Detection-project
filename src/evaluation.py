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

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    if cm.size != 4:
        raise ValueError(
            f"Confusion matrix shape {cm.shape}: both classes must be present in y_true."
        )
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true_arr, y_prob_arr)),
        "pr_auc": float(average_precision_score(y_true_arr, y_prob_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "mcc": (
            float(mcc_val)
            if not np.isnan(mcc_val := matthews_corrcoef(y_true_arr, y_pred))
            else 0.0
        ),
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


def bootstrap_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Nonparametric bootstrap CIs for headline classification metrics.

    Resamples (y_true, y_prob) pairs with replacement `n_boot` times and
    returns the central estimate plus percentile CI bounds for ROC-AUC,
    PR-AUC, F1, Brier, and MCC at a fixed threshold.

    The `y_true` sample in each bootstrap replicate is checked for both
    classes; replicates missing a class are silently skipped (rare on any
    realistic test set).
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.clip(np.asarray(y_prob).astype(float), 1e-12, 1.0 - 1e-12)
    n = len(y_true_arr)
    rng = np.random.default_rng(seed)

    roc_vals: list[float] = []
    pr_vals: list[float] = []
    f1_vals: list[float] = []
    brier_vals: list[float] = []
    mcc_vals: list[float] = []
    precision_vals: list[float] = []
    recall_vals: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true_arr[idx]
        yp = y_prob_arr[idx]
        if yt.min() == yt.max():
            continue
        ypred = (yp >= threshold).astype(int)
        roc_vals.append(roc_auc_score(yt, yp))
        pr_vals.append(average_precision_score(yt, yp))
        f1_vals.append(f1_score(yt, ypred, zero_division=0))
        brier_vals.append(brier_score_loss(yt, yp))
        mcc_vals.append(matthews_corrcoef(yt, ypred))
        precision_vals.append(precision_score(yt, ypred, zero_division=0))
        recall_vals.append(recall_score(yt, ypred, zero_division=0))

    lo = (1.0 - ci) / 2.0
    hi = 1.0 - lo

    def _summarize(name: str, vals: list[float]) -> dict[str, float]:
        arr = np.asarray(vals, dtype=float)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "ci_lo": float(np.quantile(arr, lo)),
            "ci_hi": float(np.quantile(arr, hi)),
            "std": float(arr.std(ddof=1)),
            "n_effective": int(arr.size),
        }

    return {
        "roc_auc": _summarize("roc_auc", roc_vals),
        "pr_auc": _summarize("pr_auc", pr_vals),
        "f1": _summarize("f1", f1_vals),
        "brier_loss": _summarize("brier_loss", brier_vals),
        "mcc": _summarize("mcc", mcc_vals),
        "precision": _summarize("precision", precision_vals),
        "recall": _summarize("recall", recall_vals),
    }


def reliability_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    *,
    n_bins: int = 15,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Reliability (calibration) curve as a DataFrame with bin-level stats.

    Strategy `quantile` (default) gives equal-sample bins — more stable at the
    tails than uniform binning. ECE = sum_bin (n_bin/n) * |mean_prob - mean_label|.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.clip(np.asarray(y_prob).astype(float), 1e-12, 1.0 - 1e-12)
    n = len(y_true_arr)

    if strategy == "quantile":
        edges = np.quantile(y_prob_arr, np.linspace(0, 1, n_bins + 1))
        edges[0] = 0.0
        edges[-1] = 1.0
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    # digitize into bins, clipping to valid range
    idx = np.clip(np.digitize(y_prob_arr, edges[1:-1], right=False), 0, n_bins - 1)

    rows: list[dict[str, float]] = []
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        mean_prob = float(y_prob_arr[mask].mean())
        mean_label = float(y_true_arr[mask].mean())
        gap = abs(mean_prob - mean_label)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
        rows.append(
            {
                "bin": b,
                "edge_low": float(edges[b]),
                "edge_high": float(edges[b + 1]),
                "count": cnt,
                "mean_predicted_prob": mean_prob,
                "fraction_positive": mean_label,
                "gap": gap,
            }
        )

    df = pd.DataFrame(rows)
    df.attrs["ECE"] = float(ece)
    df.attrs["MCE"] = float(mce)
    return df


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
