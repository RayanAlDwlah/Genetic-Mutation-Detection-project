#!/usr/bin/env python3
# Added for Phase 2.1 (S6): SHAP on the xgboost_phase21_esm2 model.
"""SHAP interpretation for the Phase-2.1 XGBoost model with esm2_llr.

Mirrors scripts/compute_shap.py but points at the Phase-2.1 splits and
checkpoint, and writes _phase21-suffixed artifacts. We deliberately do
NOT reuse compute_shap.py's confident-error step, because that step
joins on results/metrics/xgboost_predictions.parquet (Phase-1) and
recomputing it here would require a separate Phase-2.1 predictions
parquet that already lives at
results/metrics/phase21/xgboost_predictions.parquet from S5.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.training import prepare_split_features, select_feature_columns

warnings.filterwarnings("ignore", category=FutureWarning)

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "results/checkpoints/xgboost_phase21_esm2.ubj"
TRAIN = REPO / "data/splits/phase21/train.parquet"
VAL = REPO / "data/splits/phase21/val.parquet"
TEST = REPO / "data/splits/phase21/test.parquet"

OUT_SHAP = REPO / "results/metrics/shap_values_phase21_test.parquet"
OUT_RANK = REPO / "results/metrics/shap_ranking_phase21.csv"
OUT_SUMMARY = REPO / "results/figures/shap_summary_phase21.png"
OUT_BAR = REPO / "results/figures/shap_bar_phase21.png"
OUT_DEPENDENCE = REPO / "results/figures/shap_dependence_top3_phase21.png"

N_SAMPLE = 2_000
RNG_SEED = 42


def main() -> None:
    import shap

    rng = np.random.default_rng(RNG_SEED)
    train = pd.read_parquet(TRAIN)
    val = pd.read_parquet(VAL)
    test = pd.read_parquet(TEST)

    pos = test[test["label"] == 1]
    neg = test[test["label"] == 0]
    n_pos = min(len(pos), N_SAMPLE // 2)
    n_neg = min(len(neg), N_SAMPLE - n_pos)
    sample_idx = pd.concat(
        [pos.sample(n_pos, random_state=RNG_SEED), neg.sample(n_neg, random_state=RNG_SEED)]
    ).index
    test_sample = test.loc[sample_idx].reset_index(drop=True)

    numeric_cols, categorical_cols = select_feature_columns(train)
    _x_train, _x_val, x_test_all, feature_names = prepare_split_features(
        train, val, test, numeric_cols, categorical_cols
    )
    print(f"[shap_phase21] test sample {len(test_sample)} rows, {x_test_all.shape[1]} encoded features")
    if "num__esm2_llr" not in feature_names:
        raise SystemExit("ABORT: num__esm2_llr missing from encoded features")

    original_idx = test.index.get_indexer(sample_idx)
    x_sample = x_test_all[original_idx]
    if hasattr(x_sample, "toarray"):
        x_sample = x_sample.toarray()
    x_sample = np.asarray(x_sample, dtype=np.float32)

    booster = xgb.Booster()
    booster.load_model(str(MODEL))
    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_values(x_sample)
    if isinstance(shap_vals, list):  # binary -> 2D
        shap_vals = shap_vals[1]
    print(f"[shap_phase21] SHAP shape {shap_vals.shape}")

    # Persist values and ranking
    OUT_SHAP.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(shap_vals, columns=feature_names).to_parquet(OUT_SHAP, index=False)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    rank_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    rank_df["rank"] = rank_df.index + 1
    rank_df.to_csv(OUT_RANK, index=False)
    print("[shap_phase21] top-15 mean|SHAP|:")
    print(rank_df.head(15).to_string(index=False))

    # Plots
    import matplotlib.pyplot as plt

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_vals, features=x_sample, feature_names=feature_names, max_display=20, show=False)
    fig.tight_layout()
    fig.savefig(OUT_SUMMARY, dpi=140, bbox_inches="tight")
    plt.close(fig)

    order = np.argsort(mean_abs)[::-1][:20]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([feature_names[i] for i in order][::-1], mean_abs[order][::-1], color="#2c3e50")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("Top-20 features by mean |SHAP| --- Phase-2.1 (esm2_llr included)", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_BAR, dpi=140, bbox_inches="tight")
    plt.close(fig)

    top3 = np.argsort(mean_abs)[::-1][:3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, idx in zip(axes, top3, strict=True):
        name = feature_names[idx]
        ax.scatter(x_sample[:, idx], shap_vals[:, idx], s=8, alpha=0.5, color="#2c3e50")
        ax.axhline(0, color="gray", ls="--", lw=0.7, alpha=0.6)
        ax.set_xlabel(name)
        ax.set_ylabel("SHAP value (log-odds contribution)")
        ax.set_title(name, fontweight="bold")
        ax.grid(alpha=0.25)
    fig.suptitle("SHAP dependence --- top 3 features (Phase-2.1)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DEPENDENCE, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"[shap_phase21] wrote {OUT_SHAP}")
    print(f"[shap_phase21] wrote {OUT_RANK}")
    print(f"[shap_phase21] wrote {OUT_SUMMARY}, {OUT_BAR}, {OUT_DEPENDENCE}")

    # Locate esm2_llr in the ranking
    if "num__esm2_llr" in rank_df["feature"].values:
        rank_row = rank_df[rank_df["feature"] == "num__esm2_llr"].iloc[0]
        print(f"[shap_phase21] num__esm2_llr rank #{int(rank_row['rank'])} (mean|SHAP|={rank_row['mean_abs_shap']:.4f})")


if __name__ == "__main__":
    main()
