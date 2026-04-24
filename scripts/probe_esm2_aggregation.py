#!/usr/bin/env python3
# Added for Phase 2.1 (S1): pick the per-variant ESM-2 aggregation rule data-driven.
"""Probe four candidate aggregation rules for ESM-2 LLR per variant_key.

Variants in dbNSFP frequently map to multiple transcript isoforms; the
ESM-2 score parquet has one row per (variant_key, transcript_id) pair.
Phase 2.1 needs a single LLR per variant_key to feed XGBoost as a numeric
feature. This probe scores the test split stand-alone using each
candidate aggregation and reports ROC-AUC + average precision so the
choice is data-driven rather than imposed.

Pathogenicity = -agg(esm2_llr): higher LLR (model sees alt as plausible)
should correspond to benign, so we negate before scoring against the
binary `label` column.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
SCORES = REPO / "data/intermediate/esm2/scores_test.parquet"
TEST = REPO / "data/splits/test.parquet"
OUT = REPO / "results/metrics/esm2_aggregation_sensitivity_phase21.csv"


def aggregate(scores: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Collapse per-isoform LLR rows down to one row per variant_key."""
    scores = scores.dropna(subset=["esm2_llr"]).copy()
    if rule == "min":
        agg = scores.groupby("variant_key")["esm2_llr"].min()
    elif rule == "max":
        agg = scores.groupby("variant_key")["esm2_llr"].max()
    elif rule == "mean":
        agg = scores.groupby("variant_key")["esm2_llr"].mean()
    elif rule == "first":
        agg = scores.groupby("variant_key")["esm2_llr"].first()
    else:
        raise ValueError(f"unknown rule: {rule}")
    return agg.reset_index().rename(columns={"esm2_llr": f"esm2_llr_{rule}"})


def main() -> int:
    scores = pd.read_parquet(SCORES)
    test = pd.read_parquet(TEST)[["variant_key", "label"]]

    isoform_counts = scores.groupby("variant_key").size()
    p50 = float(isoform_counts.median())
    p95 = float(isoform_counts.quantile(0.95))
    print(f"[probe] {len(scores)} score rows over {scores['variant_key'].nunique()} unique variants "
          f"(isoforms p50={p50:.0f} p95={p95:.0f})")

    rows = []
    for rule in ["min", "max", "mean", "first"]:
        agg = aggregate(scores, rule)
        merged = test.merge(agg, on="variant_key", how="inner")
        n = len(merged)
        # Higher LLR = benign (model says alt plausible). Pathogenicity
        # score = -LLR.
        score = -merged[f"esm2_llr_{rule}"].to_numpy()
        y = merged["label"].astype(int).to_numpy()
        if y.sum() in (0, len(y)):
            print(f"[probe] {rule}: degenerate (single class)")
            continue
        roc = roc_auc_score(y, score)
        pr = average_precision_score(y, score)
        rows.append(
            {
                "rule": rule,
                "n_variants": n,
                "n_isoforms_p50": p50,
                "n_isoforms_p95": p95,
                "roc_auc": float(roc),
                "pr_auc": float(pr),
            }
        )
        print(f"[probe] {rule:5s}  n={n}  ROC={roc:.4f}  PR={pr:.4f}")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[probe] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
