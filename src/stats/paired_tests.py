"""Paired statistical tests for model-vs-model comparisons.

All functions here operate on per-sample labels and predicted scores. Each
returns a deterministic dict under a fixed seed so that the thesis numbers
are regeneratable from ``results/metrics/pairwise_pvalues.csv``.

Tests provided:
  * :func:`delong_test`       – paired ROC-AUC difference (DeLong 1988).
  * :func:`paired_bootstrap_prauc` – paired PR-AUC difference by
    non-parametric case-resampling bootstrap with a two-sided p-value.
  * :func:`auc_greater_than_half` – one-sided permutation/bootstrap test
    for ``H0: AUC <= 0.5`` (used for the denovo-db family-holdout claim).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


# ----------------------------- DeLong's test ------------------------------


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    """Fast DeLong implementation from Sun & Xu (2014); reliability-reviewed."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(
    y_true: ArrayLike,
    p1: ArrayLike,
    p2: ArrayLike,
) -> dict[str, float]:
    """Paired two-sided DeLong test for ROC-AUC(p1) - ROC-AUC(p2).

    Returns AUC estimates for each model, the paired delta, its 95% CI,
    the two-sided z-score and p-value.
    """
    y = np.asarray(y_true, dtype=int).ravel()
    p1a = np.asarray(p1, dtype=float).ravel()
    p2a = np.asarray(p2, dtype=float).ravel()
    if y.shape != p1a.shape or y.shape != p2a.shape:
        raise ValueError("y_true, p1, p2 must have identical shape")

    order = np.argsort(-y)  # positives first
    m = int(y.sum())
    preds_sorted = np.vstack([p1a[order], p2a[order]])
    aucs, cov = _fast_delong(preds_sorted, m)

    auc1, auc2 = float(aucs[0]), float(aucs[1])
    delta = auc1 - auc2
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if var <= 0:
        return {
            "auc1": auc1,
            "auc2": auc2,
            "delta": delta,
            "delta_se": 0.0,
            "delta_ci_lo": delta,
            "delta_ci_hi": delta,
            "z": float("nan"),
            "pvalue": float("nan"),
        }
    se = float(np.sqrt(var))
    z = delta / se
    pvalue = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return {
        "auc1": auc1,
        "auc2": auc2,
        "delta": delta,
        "delta_se": se,
        "delta_ci_lo": delta - 1.959964 * se,
        "delta_ci_hi": delta + 1.959964 * se,
        "z": float(z),
        "pvalue": pvalue,
    }


# ---------------------- Paired bootstrap for PR-AUC -----------------------


def paired_bootstrap_prauc(
    y_true: ArrayLike,
    p1: ArrayLike,
    p2: ArrayLike,
    *,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired case-resampling bootstrap for PR-AUC(p1) - PR-AUC(p2).

    Returns observed delta, a 95% percentile CI, and a two-sided p-value
    computed as ``2 * min(P[delta>=0], P[delta<=0])``.
    """
    y = np.asarray(y_true, dtype=int).ravel()
    p1a = np.asarray(p1, dtype=float).ravel()
    p2a = np.asarray(p2, dtype=float).ravel()
    n = len(y)
    rng = np.random.default_rng(seed)

    obs = float(average_precision_score(y, p1a) - average_precision_score(y, p2a))
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            deltas[b] = np.nan
            continue
        deltas[b] = average_precision_score(yb, p1a[idx]) - average_precision_score(
            yb, p2a[idx]
        )
    valid = deltas[~np.isnan(deltas)]
    ci_lo, ci_hi = np.percentile(valid, [2.5, 97.5])
    frac_ge = float((valid >= 0).mean())
    pvalue_two = float(2.0 * min(frac_ge, 1.0 - frac_ge))
    return {
        "prauc1": float(average_precision_score(y, p1a)),
        "prauc2": float(average_precision_score(y, p2a)),
        "delta": obs,
        "delta_ci_lo": float(ci_lo),
        "delta_ci_hi": float(ci_hi),
        "pvalue": pvalue_two,
        "n_boot_valid": int(len(valid)),
    }


# ----------------- One-sided test for AUC > chance ------------------------


def auc_greater_than_half(
    y_true: ArrayLike,
    scores: ArrayLike,
    *,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """One-sided test for ``H0: AUC <= 0.5`` vs ``H1: AUC > 0.5``.

    Combines:
      * observed AUC
      * a label-permutation p-value (exact-style null)
      * a bootstrap CI
    """
    y = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    n = len(y)
    rng = np.random.default_rng(seed)

    obs_auc = float(roc_auc_score(y, s))

    # Label-permutation p-value
    perm_aucs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        yb = rng.permutation(y)
        perm_aucs[b] = roc_auc_score(yb, s)
    pvalue_perm = float((perm_aucs >= obs_auc).mean())

    # Bootstrap CI (resample indices, preserving y,s pairing)
    boot_aucs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            boot_aucs[b] = np.nan
            continue
        boot_aucs[b] = roc_auc_score(yb, s[idx])
    valid = boot_aucs[~np.isnan(boot_aucs)]
    ci_lo, ci_hi = np.percentile(valid, [2.5, 97.5])
    return {
        "auc": obs_auc,
        "auc_ci_lo": float(ci_lo),
        "auc_ci_hi": float(ci_hi),
        "pvalue_permutation": pvalue_perm,
        "n_permutations": int(n_boot),
        "n": int(n),
        "n_positive": int(y.sum()),
    }


# ----------------- Holm-Bonferroni adjustment -----------------------------


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values in the original order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()


@dataclass
class PairwiseRow:
    """Row for ``results/metrics/pairwise_pvalues.csv``."""

    slice: str
    model_a: str
    model_b: str
    metric: str
    estimate_a: float
    estimate_b: float
    delta: float
    ci_lo: float
    ci_hi: float
    pvalue: float
    pvalue_holm: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
