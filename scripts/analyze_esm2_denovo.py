#!/usr/bin/env python3
"""Compare ESM-2 zero-shot LLR vs calibrated XGBoost on denovo-db.

The question is simple: does the protein language model's zero-shot
log-likelihood ratio carry discrimination signal on affected-vs-control
de-novo variants where the tabular baseline is at chance?

Inputs
------
- `results/metrics/esm2_denovo_db_scores.parquet`
    Per-variant ESM-2 scores + label + family_holdout flag.
- `results/metrics/external_denovo_db_predictions.parquet`
    Calibrated XGBoost probabilities on the same denovo-db sample
    (from the current Phase D run).

Outputs
-------
- `results/metrics/esm2_denovo_db_comparison.csv`
    ROC-AUC / PR-AUC point estimates + 1000-replicate bootstrap 95% CIs
    for three scores × two slices (full / family_holdout_only):
      1. XGBoost calibrated probability (baseline, tabular)
      2. ESM-2 raw LLR (zero-shot, sequence-only)
      3. XGBoost + ESM-2 rank-average fusion (cheap hybrid sanity check)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data_splitting import assign_gene_family
from src.evaluation import bootstrap_metrics


REPO = Path(__file__).resolve().parents[1]
ESM_PATH = REPO / "results/metrics/esm2_denovo_db_scores.parquet"
XGB_PRED_PATH = REPO / "results/metrics/external_denovo_db_predictions.parquet"
OUT_CSV = REPO / "results/metrics/esm2_denovo_db_comparison.csv"
TRAIN_SPLIT = REPO / "data/splits/train.parquet"


def _rank(x: np.ndarray) -> np.ndarray:
    """Convert to rank-based [0,1] scores (NaN-safe)."""
    s = pd.Series(x)
    r = s.rank(method="average", na_option="keep")
    return (r / s.notna().sum()).to_numpy()


def _score_block(y: np.ndarray, s: np.ndarray, name: str, n_boot: int = 1000) -> dict:
    ok = ~np.isnan(s)
    y_, s_ = y[ok], s[ok]
    if len(y_) < 20 or y_.min() == y_.max():
        return {"score": name, "n": int(ok.sum()), "note": "too few or single-class"}
    roc = roc_auc_score(y_, s_)
    pr = average_precision_score(y_, s_)
    # Bootstrap CIs with the project's helper so the reporting format matches
    # everything else in results/metrics/.
    boot = bootstrap_metrics(y_, s_, threshold=0.5, n_boot=n_boot, seed=42)
    return {
        "score": name, "n": int(ok.sum()), "n_pos": int(y_.sum()),
        "roc_auc": roc,
        "roc_auc_ci_lo": boot["roc_auc"]["ci_lo"],
        "roc_auc_ci_hi": boot["roc_auc"]["ci_hi"],
        "pr_auc": pr,
        "pr_auc_ci_lo": boot["pr_auc"]["ci_lo"],
        "pr_auc_ci_hi": boot["pr_auc"]["ci_hi"],
    }


def main() -> None:
    esm = pd.read_parquet(ESM_PATH)
    xgb = pd.read_parquet(XGB_PRED_PATH)

    # Resolve overlapping rows via variant_key.
    merged = esm.merge(
        xgb[["variant_key", "gene", "p_calibrated", "p_raw"]],
        on="variant_key", how="inner",
    )
    print(f"overlap: ESM-2 scored = {len(esm):,}; XGBoost predictions = "
          f"{len(xgb):,}; intersection = {len(merged):,}")

    # Family-holdout flag: gene NOT present in training family set.
    train = pd.read_parquet(TRAIN_SPLIT)
    train_fams = {assign_gene_family(g) for g in train["gene"].unique()}
    # `gene` column comes from either parquet — ESM frame has it from the
    # denovo-db loader.
    merged["family_holdout"] = ~merged["gene"].astype(str).map(assign_gene_family).isin(train_fams)
    print(f"family_holdout rows: {int(merged['family_holdout'].sum())} "
          f"({merged['family_holdout'].mean():.1%})")

    y = merged["label"].astype(int).to_numpy()
    # Scores.
    p_xgb = merged["p_calibrated"].to_numpy()
    s_esm = merged["esm2_llr"].to_numpy()
    # Cheap fusion: rank-average (scale-agnostic). Only defined where both exist.
    r_xgb = _rank(p_xgb)
    r_esm = _rank(s_esm)
    s_fuse = np.where(np.isnan(r_xgb) | np.isnan(r_esm), np.nan, (r_xgb + r_esm) / 2)

    rows: list[dict] = []
    for slice_name, mask in [("full", np.ones(len(y), dtype=bool)),
                             ("family_holdout_only", merged["family_holdout"].to_numpy())]:
        for name, s in [
            ("xgb_calibrated", p_xgb),
            ("esm2_llr", s_esm),
            ("rank_fusion_xgb_esm2", s_fuse),
        ]:
            r = _score_block(y[mask], s[mask], name)
            r["slice"] = slice_name
            rows.append(r)

    df = pd.DataFrame(rows).reindex(columns=[
        "slice", "score", "n", "n_pos",
        "roc_auc", "roc_auc_ci_lo", "roc_auc_ci_hi",
        "pr_auc", "pr_auc_ci_lo", "pr_auc_ci_hi",
        "note",
    ])
    df.to_csv(OUT_CSV, index=False)

    print()
    print(df.to_string(index=False))
    print()
    print(f"wrote {OUT_CSV.relative_to(REPO)}")


if __name__ == "__main__":
    main()
