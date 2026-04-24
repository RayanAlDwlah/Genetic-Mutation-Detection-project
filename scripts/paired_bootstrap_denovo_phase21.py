#!/usr/bin/env python3
# Added for Phase 2.1 (S9): paired bootstrap Phase-1 vs Phase-2.1 on denovo-db.
"""Paired bootstrap on denovo-db: Phase-1 (xgboost_best) vs Phase-2.1 (with ESM-2).

Reads two prediction parquets keyed on variant_key:
    Phase-1: results/metrics/external_denovo_db_predictions.parquet
    Phase-2.1: results/metrics/external_denovo_db_predictions_phase21.parquet

For each (slice in {full, holdout}, metric in {roc_auc, pr_auc}):
    - phase1_auc, phase2_auc on the inner-joined variants
    - paired_delta + 95% CI + one-sided p-value

NEVER overwrites results/metrics/denovo_paired_bootstrap.csv
(that holds the pre-vs-post-constraint paired result from P0-1).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
PHASE1 = REPO / "results/metrics/external_denovo_db_predictions.parquet"
PHASE21 = REPO / "results/metrics/external_denovo_db_predictions_phase21.parquet"
OUT = REPO / "results/metrics/denovo_paired_bootstrap_phase21.csv"


def paired_diff(y, p_a, p_b, metric_fn, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot, dtype=float)
    skipped = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if y_b.sum() in (0, n):
            diffs[b] = np.nan
            skipped += 1
            continue
        diffs[b] = metric_fn(y_b, p_b[idx]) - metric_fn(y_b, p_a[idx])
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p_one = float((diffs <= 0).mean())
    point = metric_fn(y, p_b) - metric_fn(y, p_a)
    return float(point), float(lo), float(hi), p_one, len(diffs), skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    p1 = pd.read_parquet(PHASE1)[["variant_key", "label", "p_calibrated", "family_holdout"]].rename(
        columns={"p_calibrated": "p_phase1"}
    )
    p2 = pd.read_parquet(PHASE21)[["variant_key", "p_calibrated"]].rename(
        columns={"p_calibrated": "p_phase21"}
    )
    merged = p1.merge(p2, on="variant_key", how="inner")
    print(f"[paired_bootstrap_phase21] joined {len(merged)} variants "
          f"(holdout={int(merged['family_holdout'].sum())})")
    assert len(merged) > 0, "empty join"

    rows = []
    for slice_name, mask in (
        ("full", np.ones(len(merged), dtype=bool)),
        ("holdout", merged["family_holdout"].astype(bool).to_numpy()),
    ):
        sub = merged[mask]
        y = sub["label"].astype(int).to_numpy()
        p_a = sub["p_phase1"].to_numpy()
        p_b = sub["p_phase21"].to_numpy()
        for mname, fn in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score)]:
            phase1_auc = float(fn(y, p_a))
            phase2_auc = float(fn(y, p_b))
            point, lo, hi, p_one, used, skipped = paired_diff(
                y, p_a, p_b, fn, args.n_boot, args.seed
            )
            rows.append({
                "metric": mname,
                "slice": slice_name,
                "phase1_auc": phase1_auc,
                "phase2_auc": phase2_auc,
                "paired_delta": point,
                "ci_low": lo,
                "ci_high": hi,
                "p_value_one_sided": p_one,
                "n_samples": int(mask.sum()),
                "n_boot_used": used,
                "n_boot_skipped": skipped,
            })
            sign = "+" if point >= 0 else ""
            print(f"  slice={slice_name:8s} metric={mname:7s} "
                  f"phase1={phase1_auc:.4f} phase2={phase2_auc:.4f} "
                  f"Δ={sign}{point:.4f} CI=[{lo:+.4f},{hi:+.4f}] p={p_one:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[paired_bootstrap_phase21] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
