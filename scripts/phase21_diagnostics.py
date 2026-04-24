#!/usr/bin/env python3
# Added for Phase 2.1 (S10): correlations, ablation summary, SHAP comparison plot.
"""Diagnostic artifacts for Phase-2.1 ESM-2 integration.

Generates:
  1. esm2_correlation_analysis_phase21.csv: Spearman correlation between
     esm2_llr and the conservation/constraint features it might be
     redundant with (test split).
  2. phase21_ablation_summary.csv: consolidated Δ + CI for {test PR-AUC,
     test ROC-AUC, denovo-db full ROC, denovo-db holdout ROC}.
  3. phase21_feature_importance_comparison.png: side-by-side bar chart
     of top-15 SHAP features in Phase-1 vs Phase-2.1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
TEST = REPO / "data/splits/phase21/test.parquet"
SHAP_PHASE21 = REPO / "results/metrics/shap_ranking_phase21.csv"
ABLATION_INTERNAL = REPO / "results/metrics/ablation_esm2_phase21.csv"
ABLATION_PAIRED = REPO / "results/metrics/phase21_ablation_paired_bootstrap.csv"
DENOVO_PAIRED = REPO / "results/metrics/denovo_paired_bootstrap_phase21.csv"

OUT_CORR = REPO / "results/metrics/esm2_correlation_analysis_phase21.csv"
OUT_SUMMARY = REPO / "results/metrics/phase21_ablation_summary.csv"
OUT_COMPARE = REPO / "results/figures/phase21_feature_importance_comparison.png"

PARTNERS = [
    "phyloP100way_vertebrate",
    "phyloP30way_mammalian",
    "phastCons100way_vertebrate",
    "phastCons30way_mammalian",
    "GERP++_RS",
    "GERP++_NR",
    "BLOSUM62_score",
    "Grantham_distance",
    "AN",
    "AC",
    "AF_popmax",
    "log_AF",
    "pLI",
    "oe_lof_upper",
    "mis_z",
    "lof_z",
    "oe_mis_upper",
]


def correlations() -> None:
    test = pd.read_parquet(TEST)
    sub = test.dropna(subset=["esm2_llr"]).copy()
    rows = []
    for col in PARTNERS:
        if col not in sub.columns:
            continue
        partner = sub[col].astype(float)
        valid = partner.notna() & sub["esm2_llr"].notna()
        if valid.sum() < 100:
            continue
        rho, p = spearmanr(sub.loc[valid, "esm2_llr"], partner[valid])
        rows.append({"partner": col, "n": int(valid.sum()), "spearman_rho": float(rho), "p_value": float(p)})
    df = pd.DataFrame(rows).sort_values("spearman_rho", key=abs, ascending=False).reset_index(drop=True)
    OUT_CORR.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CORR, index=False)
    print(f"[diag] wrote {OUT_CORR}")
    print(df.to_string(index=False))


def summary() -> None:
    rows: list[dict] = []
    paired_internal = pd.read_csv(ABLATION_PAIRED)
    for _, r in paired_internal.iterrows():
        rows.append({
            "comparison": "internal_test (full vs no_esm2)",
            "metric": r["metric"],
            "delta": r["point_delta"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
            "p_value": r["p_value_two_sided"],
            "test": "paired bootstrap (1000 reps)",
            "n": r["n_samples"],
        })
    paired_denovo = pd.read_csv(DENOVO_PAIRED)
    for _, r in paired_denovo.iterrows():
        rows.append({
            "comparison": f"denovo_db_{r['slice']} (Phase-2.1 vs Phase-1)",
            "metric": r["metric"],
            "delta": r["paired_delta"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
            "p_value": r["p_value_one_sided"],
            "test": "paired bootstrap (1000 reps, one-sided)",
            "n": r["n_samples"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_SUMMARY, index=False)
    print(f"[diag] wrote {OUT_SUMMARY}")
    print(df.to_string(index=False))


def shap_comparison_plot() -> None:
    import matplotlib.pyplot as plt
    p2 = pd.read_csv(SHAP_PHASE21).head(15)
    # Phase-1 SHAP ranking is in results/metrics/shap_values_test.parquet (per-row); recompute mean|SHAP|
    p1_shap = REPO / "results/metrics/shap_values_test.parquet"
    if p1_shap.exists():
        df1 = pd.read_parquet(p1_shap)
        p1 = (
            df1.abs().mean()
            .reset_index()
            .rename(columns={"index": "feature", 0: "mean_abs_shap"})
            .sort_values("mean_abs_shap", ascending=False)
            .head(15)
            .reset_index(drop=True)
        )
    else:
        p1 = pd.DataFrame({"feature": [], "mean_abs_shap": []})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, df, label in [(axes[0], p1, "Phase-1 (no ESM-2)"), (axes[1], p2, "Phase-2.1 (with ESM-2)")]:
        if df.empty:
            ax.text(0.5, 0.5, "Phase-1 SHAP parquet missing", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(label, fontweight="bold")
            continue
        ax.barh(df["feature"][::-1], df["mean_abs_shap"][::-1], color="#2c3e50")
        ax.set_xlabel("mean(|SHAP value|)")
        ax.set_title(label, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Top-15 features by mean |SHAP| --- Phase-1 vs Phase-2.1", fontweight="bold", y=1.02)
    fig.tight_layout()
    OUT_COMPARE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_COMPARE, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag] wrote {OUT_COMPARE}")


def main() -> int:
    correlations()
    summary()
    shap_comparison_plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
