#!/usr/bin/env python3
"""Ablation study: quantify contribution of allele-frequency features.

AF-based features (AF_popmax, AN, AC, log_AF) are partially correlated with
label because pathogenic variants are rarer by construction. This ablation
retrains the model with AF features removed and reports the AUC drop. A small
drop confirms the remaining ROC-AUC is not an AF-memorization artifact.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import bootstrap_metrics, compute_classification_metrics
from src.models.xgboost_model import XGBTuningConfig, tune_xgboost
from src.training import prepare_split_features, select_feature_columns
from src.utils import require_file, resolve_path


REPO_ROOT = Path(__file__).resolve().parents[1]
AF_FEATURES = {"AF", "AF_popmax", "AN", "AC", "log_AF", "is_common"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/splits/train.parquet")
    ap.add_argument("--val", default="data/splits/val.parquet")
    ap.add_argument("--test", default="data/splits/test.parquet")
    ap.add_argument("--out", default="results/metrics/ablation_af.csv")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=500)
    return ap.parse_args()


def train_and_eval(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    trials: int,
    seed: int,
    n_boot: int,
    label: str,
) -> dict[str, float]:
    x_train, x_val, x_test, _ = prepare_split_features(
        train_df, val_df, test_df, numeric_cols, categorical_cols
    )
    y_train = train_df["label"].astype(int)
    y_val = val_df["label"].astype(int)
    y_test = test_df["label"].astype(int)
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = float(neg / max(pos, 1))

    cfg = XGBTuningConfig(n_trials=max(2, trials), seed=seed, n_estimators=2500, early_stopping_rounds=80)
    model, _, _ = tune_xgboost(x_train, y_train, x_val, y_val, config=cfg, scale_pos_weight=spw)

    p_test = model.predict_proba(x_test)[:, 1]
    metrics = compute_classification_metrics(y_test, p_test, threshold=0.5)
    boot = bootstrap_metrics(y_test.to_numpy(), p_test, threshold=0.5, n_boot=n_boot, seed=seed)

    return {
        "variant": label,
        "n_features": len(numeric_cols) + len(categorical_cols),
        "test_roc_auc": metrics["roc_auc"],
        "test_roc_auc_ci_lo": boot["roc_auc"]["ci_lo"],
        "test_roc_auc_ci_hi": boot["roc_auc"]["ci_hi"],
        "test_pr_auc": metrics["pr_auc"],
        "test_pr_auc_ci_lo": boot["pr_auc"]["ci_lo"],
        "test_pr_auc_ci_hi": boot["pr_auc"]["ci_hi"],
        "test_brier": metrics["brier_loss"],
        "test_f1": metrics["f1"],
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_df = pd.read_parquet(resolve_path(REPO_ROOT, args.train))
    val_df = pd.read_parquet(resolve_path(REPO_ROOT, args.val))
    test_df = pd.read_parquet(resolve_path(REPO_ROOT, args.test))

    numeric_cols, categorical_cols = select_feature_columns(train_df)

    rows: list[dict[str, float]] = []

    # Full model (baseline reference)
    print("=" * 60)
    print("Training FULL model (reference)…")
    rows.append(
        train_and_eval(
            train_df, val_df, test_df, numeric_cols, categorical_cols,
            args.trials, args.seed, args.n_boot, "full",
        )
    )

    # No-AF variant
    numeric_noaf = [c for c in numeric_cols if c not in AF_FEATURES]
    categorical_noaf = [c for c in categorical_cols if c not in AF_FEATURES]
    print()
    print("=" * 60)
    print(f"Training NO-AF model ({len(AF_FEATURES)} features removed)…")
    rows.append(
        train_and_eval(
            train_df, val_df, test_df, numeric_noaf, categorical_noaf,
            args.trials, args.seed, args.n_boot, "no_af",
        )
    )

    # No-conservation variant (sanity check — should hurt a lot)
    cons_keywords = ("phyloP", "phastCons", "GERP", "SiPhy")
    numeric_nocons = [c for c in numeric_cols if not any(k in c for k in cons_keywords)]
    print()
    print("=" * 60)
    print("Training NO-CONSERVATION model (sanity check — expected big drop)…")
    rows.append(
        train_and_eval(
            train_df, val_df, test_df, numeric_nocons, categorical_cols,
            args.trials, args.seed, args.n_boot, "no_conservation",
        )
    )

    # No AA-properties variant
    aa_keywords = ("hydrophobicity", "molecular_weight", "pI_", "volume", "polarity", "charge", "Grantham", "BLOSUM")
    numeric_noaa = [c for c in numeric_cols if not any(k in c for k in aa_keywords)]
    print()
    print("=" * 60)
    print("Training NO-AA-PROPS model…")
    rows.append(
        train_and_eval(
            train_df, val_df, test_df, numeric_noaa, categorical_cols,
            args.trials, args.seed, args.n_boot, "no_aa_props",
        )
    )

    df = pd.DataFrame(rows)
    out = resolve_path(REPO_ROOT, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print()
    print("=" * 60)
    print("Ablation summary (test set):")
    print(df.to_string(index=False))
    print()
    print(f"Saved: {out}")

    # Compute deltas vs full
    full = df[df["variant"] == "full"].iloc[0]
    print()
    print("Delta vs FULL baseline (test):")
    for _, row in df.iterrows():
        if row["variant"] == "full":
            continue
        d_roc = row["test_roc_auc"] - full["test_roc_auc"]
        d_pr = row["test_pr_auc"] - full["test_pr_auc"]
        d_f1 = row["test_f1"] - full["test_f1"]
        print(f"  {row['variant']:20s} ΔROC={d_roc:+.4f}  ΔPR={d_pr:+.4f}  ΔF1={d_f1:+.4f}")


if __name__ == "__main__":
    main()
