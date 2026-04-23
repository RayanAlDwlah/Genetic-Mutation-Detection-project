#!/usr/bin/env python3
"""Rank-fusion proof-of-concept: XGBoost × ESM-2 LLR on the paralog-disjoint test.

Motivation
----------
While the full ClinVar training-set ESM-2 scoring is still pending a GPU-quota
reset, we already have per-variant LLR scores for val + test (≈ 56 K variants,
99.4 % coverage). That's enough to run a clean rank-fusion experiment:

  * XGBoost alone (our supervised baseline)
  * ESM-2 LLR alone (zero-shot unsupervised)
  * A simple late-fusion score that averages the two per-variant ranks
  * Weight-tuned fusion where the weight α ∈ [0,1] on the ESM-2 rank is
    chosen on **val** and then frozen for the **test** evaluation.

This is identical in spirit to what we did for the denovo-db holdout
earlier (+0.015 ROC). If the same pattern holds here, we expect a
slightly larger gain because the ClinVar test set is cleaner (no
cross-dataset build-coordinate issues).

Inputs
------
- ``results/metrics/xgboost_predictions.parquet``
    (variant_key, split, y_true, p_raw, p_calibrated)
- ``data/intermediate/esm2/scores_val.parquet``
- ``data/intermediate/esm2/scores_test.parquet``
    Both contain esm2_llr among other columns.

Outputs
-------
- ``results/metrics/rank_fusion_esm2.csv``  — metrics table
- ``results/figures/rank_fusion_esm2.png``  — side-by-side ROC + PR curves

Reproducibility
---------------
Deterministic: no randomness outside numpy/sklearn, no model training.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

REPO = Path(__file__).resolve().parents[1]
PRED_PATH = REPO / "results/metrics/xgboost_predictions.parquet"
ESM_DIR = REPO / "data/intermediate/esm2"
OUT_CSV = REPO / "results/metrics/rank_fusion_esm2.csv"
OUT_FIG = REPO / "results/figures/rank_fusion_esm2.png"


def _load_scores() -> pd.DataFrame:
    """Merge XGBoost predictions with ESM-2 LLR for val + test."""
    pred = pd.read_parquet(PRED_PATH)
    val_esm = pd.read_parquet(ESM_DIR / "scores_val.parquet")[
        ["variant_key", "esm2_llr"]
    ]
    test_esm = pd.read_parquet(ESM_DIR / "scores_test.parquet")[
        ["variant_key", "esm2_llr"]
    ]
    esm = pd.concat([val_esm, test_esm], ignore_index=True).drop_duplicates("variant_key")
    merged = pred.merge(esm, on="variant_key", how="left")
    print(
        f"[rank-fusion] merged {len(merged):,} predictions "
        f"({merged['esm2_llr'].notna().sum():,} with ESM-2 LLR, "
        f"{merged['esm2_llr'].isna().sum():,} missing)"
    )
    return merged


def _to_rank(x: pd.Series) -> np.ndarray:
    """Convert raw scores to ranks in [0, 1]. Higher score → higher rank.

    NaN scores receive the median rank so they neither help nor hurt
    the fused score.
    """
    arr = x.to_numpy(dtype=float)
    mask = ~np.isnan(arr)
    ranks = np.full_like(arr, fill_value=np.nan)
    # Dense rank then normalize.
    order = arr[mask].argsort()
    rk = np.empty_like(order)
    rk[order] = np.arange(mask.sum())
    ranks[mask] = rk / max(1, mask.sum() - 1)
    # Missing → median rank (0.5) so the fused score is neutral on missing.
    ranks[~mask] = 0.5
    return ranks


def _fuse(xgb_rank: np.ndarray, esm_rank: np.ndarray, alpha: float) -> np.ndarray:
    """Weight α goes on ESM-2; (1 − α) on XGBoost."""
    return (1.0 - alpha) * xgb_rank + alpha * esm_rank


def _metrics(y: np.ndarray, s: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
    }


def main() -> None:
    merged = _load_scores()
    train_mask = merged["split"] == "val"  # used here only to tune α
    test_mask = merged["split"] == "test"

    # ── 1. Per-split raw arrays ────────────────────────────────────────
    val = merged.loc[train_mask].copy()
    test = merged.loc[test_mask].copy()

    # Sign convention: our LLR = log P(alt) - log P(ref). Higher LLR means
    # the model finds the alternate allele *plausible* in context → the
    # variant is more likely tolerated/benign. Pathogenicity scales with
    # −LLR (the more implausible the alt, the more likely it disrupts
    # function). This matches Brandes (2023)'s ESM-1b zero-shot sign
    # convention. We flip once here so every downstream rank / metric
    # treats "higher = more pathogenic" uniformly.
    val["esm2_path"] = -val["esm2_llr"]
    test["esm2_path"] = -test["esm2_llr"]

    val["xgb_rank"] = _to_rank(val["p_calibrated"])
    val["esm_rank"] = _to_rank(val["esm2_path"])
    test["xgb_rank"] = _to_rank(test["p_calibrated"])
    test["esm_rank"] = _to_rank(test["esm2_path"])

    # ── 2. Tune α on val (PR-AUC objective) ────────────────────────────
    alphas = np.linspace(0.0, 1.0, 41)
    best_alpha = 0.0
    best_val_pr = -np.inf
    sweep = []
    for a in alphas:
        s = _fuse(val["xgb_rank"].to_numpy(), val["esm_rank"].to_numpy(), a)
        pr = average_precision_score(val["y_true"], s)
        roc = roc_auc_score(val["y_true"], s)
        sweep.append({"alpha": float(a), "val_roc": float(roc), "val_pr": float(pr)})
        if pr > best_val_pr:
            best_val_pr = float(pr)
            best_alpha = float(a)
    print(f"[rank-fusion] tuned α = {best_alpha:.2f} (val PR-AUC = {best_val_pr:.4f})")

    # ── 3. Evaluate all models on test ─────────────────────────────────
    y_test = test["y_true"].to_numpy()
    rows = []

    # XGBoost alone (both raw and calibrated — we report calibrated as the baseline).
    rows.append({"model": "xgboost_calibrated", **_metrics(y_test, test["p_calibrated"].to_numpy())})
    rows.append(
        {
            "model": "esm2_llr_zero_shot",
            **_metrics(y_test, test["esm2_path"].fillna(test["esm2_path"].median()).to_numpy()),
        }
    )
    rows.append(
        {
            "model": "rank_fusion_uniform",
            **_metrics(y_test, _fuse(test["xgb_rank"].to_numpy(), test["esm_rank"].to_numpy(), 0.5)),
        }
    )
    rows.append(
        {
            "model": "rank_fusion_tuned",
            **_metrics(
                y_test, _fuse(test["xgb_rank"].to_numpy(), test["esm_rank"].to_numpy(), best_alpha)
            ),
            "alpha": best_alpha,
        }
    )
    table = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    print(f"[rank-fusion] wrote {OUT_CSV}")
    print(table.to_string(index=False))

    # ── 4. Figure: ROC + PR side-by-side on test ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    palette = {
        "xgboost_calibrated": "#045531",
        "esm2_llr_zero_shot": "#d97706",
        "rank_fusion_uniform": "#7c3aed",
        "rank_fusion_tuned": "#2563eb",
    }
    labels = {
        "xgboost_calibrated": "XGBoost (calibrated)",
        "esm2_llr_zero_shot": "ESM-2 LLR (zero-shot)",
        "rank_fusion_uniform": "Rank fusion (α = 0.5)",
        "rank_fusion_tuned": f"Rank fusion (α = {best_alpha:.2f}, tuned on val)",
    }
    score_map = {
        "xgboost_calibrated": test["p_calibrated"].to_numpy(),
        "esm2_llr_zero_shot": test["esm2_path"].fillna(test["esm2_path"].median()).to_numpy(),
        "rank_fusion_uniform": _fuse(test["xgb_rank"].to_numpy(), test["esm_rank"].to_numpy(), 0.5),
        "rank_fusion_tuned": _fuse(
            test["xgb_rank"].to_numpy(), test["esm_rank"].to_numpy(), best_alpha
        ),
    }

    for name, s in score_map.items():
        fpr, tpr, _ = roc_curve(y_test, s)
        prec, rec, _ = precision_recall_curve(y_test, s)
        roc = roc_auc_score(y_test, s)
        pr = average_precision_score(y_test, s)
        axes[0].plot(fpr, tpr, color=palette[name], label=f"{labels[name]}  (AUC = {roc:.3f})", lw=2)
        axes[1].plot(rec, prec, color=palette[name], label=f"{labels[name]}  (AP = {pr:.3f})", lw=2)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].set_title("ROC on paralog-disjoint test")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(alpha=0.25)

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall on paralog-disjoint test")
    axes[1].legend(loc="lower left", fontsize=9)
    axes[1].grid(alpha=0.25)

    fig.suptitle(
        "Rank-fusion proof-of-concept: XGBoost × ESM-2 LLR\n"
        f"α tuned on val ({best_alpha:.2f}), evaluated on held-out test ({len(test):,} variants)",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"[rank-fusion] wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
