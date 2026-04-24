#!/usr/bin/env python3
# Added for the P0 revision pass (see CLAUDE_CODE_P0_FIXES.md, P0-1).
"""Paired bootstrap test for denovo-db pre/post-constraint improvement."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "results" / "metrics"


def load_paired_predictions(slice_name: str = "holdout") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, p_pre, p_post) aligned on variant_id, filtered to the slice."""
    pre = pd.read_parquet(METRICS_DIR / "denovo_predictions_pre_constraint.parquet")
    post = pd.read_parquet(METRICS_DIR / "denovo_predictions_post_constraint.parquet")
    if "slice" in pre.columns:
        pre = pre.loc[pre["slice"] == slice_name]
        post = post.loc[post["slice"] == slice_name]
    merged = pre.merge(
        post[["variant_id", "p_pred"]],
        on="variant_id",
        how="inner",
        suffixes=("_pre", "_post"),
    )
    assert len(merged) > 0, "Empty merge. Check variant_id alignment."
    assert (merged["y_true"].isin([0, 1])).all()
    return (
        merged["y_true"].to_numpy(),
        merged["p_pred_pre"].to_numpy(),
        merged["p_pred_post"].to_numpy(),
    )


def paired_bootstrap_delta(
    y_true: np.ndarray,
    p_pre: np.ndarray,
    p_post: np.ndarray,
    metric: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap CI on delta metric = metric(post) - metric(pre)."""
    metric_fn = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}[metric]
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot, dtype=float)
    skipped = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        # Degenerate resample (single-class) breaks AUROC
        if y_b.sum() in (0, n):
            skipped += 1
            deltas[b] = np.nan
            continue
        deltas[b] = metric_fn(y_b, p_post[idx]) - metric_fn(y_b, p_pre[idx])
    valid = ~np.isnan(deltas) & np.isfinite(deltas)
    deltas = deltas[valid]
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    # One-sided p-value for H0: delta <= 0
    p_value = float((deltas <= 0).mean())
    # Point estimate on the full sample
    point = metric_fn(y_true, p_post) - metric_fn(y_true, p_pre)
    return {
        "metric": metric,
        "point_delta": float(point),
        "bootstrap_mean_delta": float(deltas.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value_one_sided": p_value,
        "n_samples": int(n),
        "n_boot_used": int(len(deltas)),
        "n_boot_skipped": int(skipped),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--slice",
        type=str,
        default="holdout",
        choices=["holdout", "full"],
    )
    args = parser.parse_args()

    y_true, p_pre, p_post = load_paired_predictions(args.slice)
    rows = []
    for metric in ["roc_auc", "pr_auc"]:
        result = paired_bootstrap_delta(
            y_true, p_pre, p_post, metric=metric,
            n_boot=args.n_boot, seed=args.seed,
        )
        result["slice"] = args.slice
        rows.append(result)
        print(
            f"[paired_bootstrap] slice={args.slice} metric={metric} "
            f"delta={result['point_delta']:+.4f} "
            f"CI95=[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] "
            f"p={result['p_value_one_sided']:.4f} (n_boot={result['n_boot_used']})"
        )
    out = METRICS_DIR / "denovo_paired_bootstrap.csv"
    df = pd.DataFrame(rows)
    # Append mode: keep rows from prior --slice runs on the same day.
    if out.exists():
        existing = pd.read_csv(out)
        # Drop prior rows for this slice so re-runs are idempotent.
        existing = existing[existing["slice"] != args.slice]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"[paired_bootstrap] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
