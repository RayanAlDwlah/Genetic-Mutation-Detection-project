#!/usr/bin/env python3
# Added for Phase 2.1 (S7): with vs without ESM-2 ablation, paired bootstrap on test PR-AUC.
"""Ablation study: train Phase-2.1 with vs without ESM-2 features.

Trains three identical XGBoost models (same frozen Phase-1 hyperparameters)
on the Phase-2.1 splits, varying only the feature set:
    full      : all features (= xgboost_phase21_esm2.ubj equivalent)
    no_esm2   : drop {esm2_llr, is_imputed_esm2_llr}
    esm2_only : keep {esm2_llr, is_imputed_esm2_llr, label} only

Reports point estimates + bootstrap CIs per condition AND a paired
bootstrap (1000 reps, seed=42) of Δ(test PR-AUC) and Δ(test ROC-AUC)
between full and no_esm2 --- this is the headline statistical claim
of Phase 2.1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation import bootstrap_metrics
from src.models.xgboost_model import build_xgboost_model
from src.training import prepare_split_features, select_feature_columns

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "data/splits/phase21/train.parquet"
VAL = REPO / "data/splits/phase21/val.parquet"
TEST = REPO / "data/splits/phase21/test.parquet"
PARAMS = REPO / "results/metrics/xgboost_best_params.csv"

ESM2_COLS = ["esm2_llr", "is_imputed_esm2_llr"]


def load_phase1_params() -> dict[str, float | int]:
    df = pd.read_csv(PARAMS)
    row = df.iloc[0]
    keys = [
        "max_depth", "learning_rate", "min_child_weight", "subsample",
        "colsample_bytree", "colsample_bylevel", "gamma", "reg_alpha",
        "reg_lambda", "max_delta_step", "scale_pos_weight",
    ]
    int_keys = {"max_depth", "min_child_weight", "max_delta_step"}
    out: dict[str, float | int] = {}
    for k in keys:
        if pd.notna(row[k]):
            out[k] = int(row[k]) if k in int_keys else float(row[k])
    return out


def train_one(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    drop: list[str],
    keep_only: list[str] | None,
    params: dict[str, float | int],
    scale_pos_weight: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Train one XGBoost on a (filtered) feature set; return (val_probs, test_probs, feature_names)."""
    if keep_only is not None:
        keep = set(keep_only) | {"label"}
        train_use = train[[c for c in train.columns if c in keep]].copy()
        val_use = val[[c for c in val.columns if c in keep]].copy()
        test_use = test[[c for c in test.columns if c in keep]].copy()
    else:
        train_use = train.drop(columns=[c for c in drop if c in train.columns]).copy()
        val_use = val.drop(columns=[c for c in drop if c in val.columns]).copy()
        test_use = test.drop(columns=[c for c in drop if c in test.columns]).copy()

    numeric_cols, categorical_cols = select_feature_columns(train_use)
    x_train, x_val, x_test, feat = prepare_split_features(
        train_use, val_use, test_use, numeric_cols, categorical_cols
    )
    y_train = train_use["label"].astype(int)
    y_val = val_use["label"].astype(int)
    p = dict(params)
    p["scale_pos_weight"] = scale_pos_weight
    model = build_xgboost_model(p, seed=seed, n_estimators=2500, early_stopping_rounds=80)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    p_val = model.predict_proba(x_val)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]
    return p_val, p_test, feat


def paired_bootstrap_diff(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Δ = metric(p_a) − metric(p_b) on the SAME resampled indices (paired)."""
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
        diffs[b] = metric_fn(y_b, p_a[idx]) - metric_fn(y_b, p_b[idx])
    valid = ~np.isnan(diffs) & np.isfinite(diffs)
    diffs = diffs[valid]
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    point = metric_fn(y, p_a) - metric_fn(y, p_b)
    return {
        "point_delta": float(point),
        "boot_mean_delta": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value_two_sided": float(p_two),
        "n_samples": int(n),
        "n_boot_used": int(len(diffs)),
        "n_boot_skipped": int(skipped),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train = pd.read_parquet(TRAIN)
    val = pd.read_parquet(VAL)
    test = pd.read_parquet(TEST)
    y_test = test["label"].astype(int).to_numpy()

    pos = int((train["label"] == 1).sum())
    neg = int((train["label"] == 0).sum())
    spw = float(neg / max(pos, 1))
    params = load_phase1_params()
    print(f"[ablate_esm2] frozen params: {params}")

    rows = []
    probs_test: dict[str, np.ndarray] = {}
    for cond_name, drop, keep_only in [
        ("full", [], None),
        ("no_esm2", ESM2_COLS, None),
        ("esm2_only", [], ESM2_COLS),
    ]:
        print(f"\n[ablate_esm2] training condition={cond_name}")
        p_val, p_test, feat = train_one(train, val, test, drop, keep_only, params, spw, args.seed)
        probs_test[cond_name] = p_test
        roc_boot = bootstrap_metrics(y_test, p_test, threshold=0.5, n_boot=args.n_boot, seed=args.seed)
        rows.append({
            "condition": cond_name,
            "n_features": len(feat),
            "test_roc_auc": float(roc_auc_score(y_test, p_test)),
            "test_pr_auc": float(average_precision_score(y_test, p_test)),
            "test_roc_ci_low": roc_boot["roc_auc"]["ci_lo"],
            "test_roc_ci_high": roc_boot["roc_auc"]["ci_hi"],
            "test_pr_ci_low": roc_boot["pr_auc"]["ci_lo"],
            "test_pr_ci_high": roc_boot["pr_auc"]["ci_hi"],
        })
        print(f"  ROC={rows[-1]['test_roc_auc']:.4f} PR={rows[-1]['test_pr_auc']:.4f} "
              f"n_features={len(feat)}")

    out_df = pd.DataFrame(rows)
    OUT_AB = REPO / "results/metrics/ablation_esm2_phase21.csv"
    OUT_AB.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_AB, index=False)
    print(f"\n[ablate_esm2] wrote {OUT_AB}")
    print(out_df.to_string(index=False))

    # Paired bootstrap full vs no_esm2
    print("\n[ablate_esm2] paired bootstrap full vs no_esm2 (test split, n=%d)..." % len(y_test))
    pair_rows = []
    for metric_name, fn in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score)]:
        res = paired_bootstrap_diff(y_test, probs_test["full"], probs_test["no_esm2"],
                                    metric_fn=fn, n_boot=args.n_boot, seed=args.seed)
        res["metric"] = metric_name
        res["comparison"] = "full vs no_esm2"
        pair_rows.append(res)
        print(f"  Δ{metric_name}={res['point_delta']:+.4f}  "
              f"CI95=[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]  "
              f"p={res['p_value_two_sided']:.4f}")
    OUT_PB = REPO / "results/metrics/phase21_ablation_paired_bootstrap.csv"
    pd.DataFrame(pair_rows).to_csv(OUT_PB, index=False)
    print(f"[ablate_esm2] wrote {OUT_PB}")

    # Forest plot
    import matplotlib.pyplot as plt

    phase1_test_pr = 0.8273  # canonical Phase-1 calibrated PR-AUC
    phase1_test_pr_lo, phase1_test_pr_hi = 0.8185, 0.8352
    phase1_test_roc = 0.9376
    phase1_test_roc_lo, phase1_test_roc_hi = 0.9346, 0.9404

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ["Phase-1 (no ESM-2)", "Phase-2.1 no_esm2", "Phase-2.1 esm2_only", "Phase-2.1 full"]

    pr_means = [
        phase1_test_pr,
        out_df.loc[out_df.condition == "no_esm2", "test_pr_auc"].iloc[0],
        out_df.loc[out_df.condition == "esm2_only", "test_pr_auc"].iloc[0],
        out_df.loc[out_df.condition == "full", "test_pr_auc"].iloc[0],
    ]
    pr_los = [
        phase1_test_pr_lo,
        out_df.loc[out_df.condition == "no_esm2", "test_pr_ci_low"].iloc[0],
        out_df.loc[out_df.condition == "esm2_only", "test_pr_ci_low"].iloc[0],
        out_df.loc[out_df.condition == "full", "test_pr_ci_low"].iloc[0],
    ]
    pr_his = [
        phase1_test_pr_hi,
        out_df.loc[out_df.condition == "no_esm2", "test_pr_ci_high"].iloc[0],
        out_df.loc[out_df.condition == "esm2_only", "test_pr_ci_high"].iloc[0],
        out_df.loc[out_df.condition == "full", "test_pr_ci_high"].iloc[0],
    ]
    roc_means = [
        phase1_test_roc,
        out_df.loc[out_df.condition == "no_esm2", "test_roc_auc"].iloc[0],
        out_df.loc[out_df.condition == "esm2_only", "test_roc_auc"].iloc[0],
        out_df.loc[out_df.condition == "full", "test_roc_auc"].iloc[0],
    ]
    roc_los = [
        phase1_test_roc_lo,
        out_df.loc[out_df.condition == "no_esm2", "test_roc_ci_low"].iloc[0],
        out_df.loc[out_df.condition == "esm2_only", "test_roc_ci_low"].iloc[0],
        out_df.loc[out_df.condition == "full", "test_roc_ci_low"].iloc[0],
    ]
    roc_his = [
        phase1_test_roc_hi,
        out_df.loc[out_df.condition == "no_esm2", "test_roc_ci_high"].iloc[0],
        out_df.loc[out_df.condition == "esm2_only", "test_roc_ci_high"].iloc[0],
        out_df.loc[out_df.condition == "full", "test_roc_ci_high"].iloc[0],
    ]
    ys = list(range(len(labels)))
    for ax, means, los, his, name in [
        (axes[0], pr_means, pr_los, pr_his, "Test PR-AUC"),
        (axes[1], roc_means, roc_los, roc_his, "Test ROC-AUC"),
    ]:
        ax.errorbar(
            means, ys,
            xerr=[[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]],
            fmt="o", color="#2c3e50", ecolor="#7f8c8d", capsize=4,
        )
        ax.set_yticks(ys)
        ax.set_yticklabels(labels)
        ax.set_xlabel(name)
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
    fig.suptitle("Phase-2.1 ESM-2 ablation (test split, calibrated)", fontweight="bold", y=1.02)
    fig.tight_layout()
    OUT_FOREST = REPO / "results/figures/ablation_esm2_forest_phase21.png"
    OUT_FOREST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FOREST, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[ablate_esm2] wrote {OUT_FOREST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
