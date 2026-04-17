#!/usr/bin/env python3
"""Professional-grade evaluation suite for the XGBoost baseline.

Produces:
  - Bootstrap 95% CIs for ROC-AUC / PR-AUC / F1 / Brier / MCC
  - Reliability curve + ECE/MCE (raw & isotonic-calibrated)
  - Clinical operating points (precision @ recall targets, recall @ precision targets)
  - Isotonic-calibrated probability predictions saved to parquet

All metrics reported on validation and test splits; calibrator is fit on
validation only. This module is idempotent — rerunning rebuilds all artifacts
without retraining the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve

from src.evaluation import (
    bootstrap_metrics,
    compute_classification_metrics,
    reliability_curve,
)
from src.training import prepare_split_features, select_feature_columns
from src.utils import require_file, resolve_path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "results/checkpoints/xgboost_best.ubj"
DEFAULT_TRAIN = "data/splits/train.parquet"
DEFAULT_VAL = "data/splits/val.parquet"
DEFAULT_TEST = "data/splits/test.parquet"
DEFAULT_PARAMS = "results/metrics/xgboost_best_params.csv"
DEFAULT_OUT_DIR = "results/metrics"

RECALL_TARGETS = (0.80, 0.90, 0.95, 0.99)
PRECISION_TARGETS = (0.80, 0.90, 0.95, 0.99)


def _predict(model: xgb.Booster, X) -> np.ndarray:
    d = xgb.DMatrix(X)
    return model.predict(d)


def _operating_points(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Find thresholds that hit specific recall / precision targets."""
    precision, recall, thresh = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thresh) == len(precision)-1; align.
    # Truncate precision/recall to match thresh length.
    p, r, t = precision[:-1], recall[:-1], thresh

    rows: list[dict[str, float]] = []
    for target in RECALL_TARGETS:
        feasible = r >= target
        if feasible.any():
            # Pick the threshold maximizing precision subject to recall >= target
            idx = np.argmax(np.where(feasible, p, -np.inf))
            rows.append(
                {
                    "constraint": f"recall>={target:.2f}",
                    "threshold": float(t[idx]),
                    "precision": float(p[idx]),
                    "recall": float(r[idx]),
                    "achieved": True,
                }
            )
        else:
            rows.append(
                {
                    "constraint": f"recall>={target:.2f}",
                    "threshold": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "achieved": False,
                }
            )

    for target in PRECISION_TARGETS:
        feasible = p >= target
        if feasible.any():
            idx = np.argmax(np.where(feasible, r, -np.inf))
            rows.append(
                {
                    "constraint": f"precision>={target:.2f}",
                    "threshold": float(t[idx]),
                    "precision": float(p[idx]),
                    "recall": float(r[idx]),
                    "achieved": True,
                }
            )
        else:
            rows.append(
                {
                    "constraint": f"precision>={target:.2f}",
                    "threshold": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "achieved": False,
                }
            )

    return pd.DataFrame(rows)


def _format_boot(name: str, boot: dict[str, dict[str, float]]) -> dict[str, float]:
    """Flatten nested bootstrap dict into a single-row record."""
    row: dict[str, float] = {"metric_set": name}
    for metric, stats in boot.items():
        for k, v in stats.items():
            row[f"{metric}__{k}"] = v
    return row


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bootstrap CIs, calibration, operating points")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train", default=DEFAULT_TRAIN)
    ap.add_argument("--val", default=DEFAULT_VAL)
    ap.add_argument("--test", default=DEFAULT_TEST)
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_path(REPO_ROOT, args.model)
    train_path = resolve_path(REPO_ROOT, args.train)
    val_path = resolve_path(REPO_ROOT, args.val)
    test_path = resolve_path(REPO_ROOT, args.test)
    params_path = resolve_path(REPO_ROOT, args.params)
    out_dir = resolve_path(REPO_ROOT, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in (model_path, train_path, val_path, test_path, params_path):
        require_file(p)

    # Load data & rebuild feature matrices with the SAME preprocessing as training.
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    numeric_cols, categorical_cols = select_feature_columns(train_df)
    x_train, x_val, x_test, _ = prepare_split_features(
        train_df, val_df, test_df, numeric_cols, categorical_cols
    )
    y_val = val_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    # Load model and best threshold from params CSV.
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    params_df = pd.read_csv(params_path)
    best_threshold = float(params_df.iloc[0]["best_threshold"])

    # Raw probabilities
    p_val_raw = _predict(booster, x_val)
    p_test_raw = _predict(booster, x_test)

    # Isotonic calibration fit on validation predictions
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_val_raw, y_val)
    p_val_cal = iso.transform(p_val_raw)
    p_test_cal = iso.transform(p_test_raw)

    # ─── Bootstrap CIs ───────────────────────────────────────────────────────
    print(f"Bootstrapping {args.n_boot} replicates for val+test (raw & calibrated)…")
    boot_rows = []
    for name, y, p in [
        ("val_raw", y_val, p_val_raw),
        ("val_calibrated", y_val, p_val_cal),
        ("test_raw", y_test, p_test_raw),
        ("test_calibrated", y_test, p_test_cal),
    ]:
        boot = bootstrap_metrics(y, p, threshold=best_threshold, n_boot=args.n_boot, seed=args.seed)
        boot_rows.append(_format_boot(name, boot))
    boot_df = pd.DataFrame(boot_rows)
    boot_out = out_dir / "xgboost_bootstrap_ci.csv"
    boot_df.to_csv(boot_out, index=False)
    print(f"  saved {boot_out}")

    # ─── Reliability curves ──────────────────────────────────────────────────
    rel_rows = []
    ece_summary = []
    for name, y, p in [
        ("val_raw", y_val, p_val_raw),
        ("val_calibrated", y_val, p_val_cal),
        ("test_raw", y_test, p_test_raw),
        ("test_calibrated", y_test, p_test_cal),
    ]:
        curve = reliability_curve(y, p, n_bins=15, strategy="quantile")
        curve.insert(0, "eval_set", name)
        rel_rows.append(curve)
        ece_summary.append(
            {"eval_set": name, "ECE": curve.attrs["ECE"], "MCE": curve.attrs["MCE"]}
        )
    rel_df = pd.concat(rel_rows, ignore_index=True)
    rel_out = out_dir / "xgboost_reliability_curve.csv"
    rel_df.to_csv(rel_out, index=False)
    ece_df = pd.DataFrame(ece_summary)
    ece_out = out_dir / "xgboost_calibration_summary.csv"
    ece_df.to_csv(ece_out, index=False)
    print(f"  saved {rel_out}")
    print(f"  saved {ece_out}")

    # ─── Operating points ────────────────────────────────────────────────────
    op_parts = []
    for name, y, p in [
        ("val_raw", y_val, p_val_raw),
        ("val_calibrated", y_val, p_val_cal),
        ("test_raw", y_test, p_test_raw),
        ("test_calibrated", y_test, p_test_cal),
    ]:
        df = _operating_points(y, p)
        df.insert(0, "eval_set", name)
        op_parts.append(df)
    op_df = pd.concat(op_parts, ignore_index=True)
    op_out = out_dir / "xgboost_operating_points.csv"
    op_df.to_csv(op_out, index=False)
    print(f"  saved {op_out}")

    # ─── Point-estimate metrics (calibrated) ─────────────────────────────────
    cal_rows = []
    for name, y, p in [
        ("val_raw", y_val, p_val_raw),
        ("val_calibrated", y_val, p_val_cal),
        ("test_raw", y_test, p_test_raw),
        ("test_calibrated", y_test, p_test_cal),
    ]:
        m = compute_classification_metrics(y, p, threshold=best_threshold)
        m["eval_set"] = name
        cal_rows.append(m)
    cal_df = pd.DataFrame(cal_rows)
    cols = ["eval_set"] + [c for c in cal_df.columns if c != "eval_set"]
    cal_df = cal_df[cols]
    cal_out = out_dir / "xgboost_calibrated_metrics.csv"
    cal_df.to_csv(cal_out, index=False)
    print(f"  saved {cal_out}")

    # ─── Calibrated probabilities for downstream use ─────────────────────────
    prob_df = pd.DataFrame(
        {
            "variant_key": pd.concat([val_df["variant_key"], test_df["variant_key"]], ignore_index=True),
            "split": ["val"] * len(val_df) + ["test"] * len(test_df),
            "y_true": np.concatenate([y_val, y_test]),
            "p_raw": np.concatenate([p_val_raw, p_test_raw]),
            "p_calibrated": np.concatenate([p_val_cal, p_test_cal]),
        }
    )
    prob_out = out_dir / "xgboost_predictions.parquet"
    prob_df.to_parquet(prob_out, index=False)
    print(f"  saved {prob_out}")

    # ─── Console summary ─────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"TEST SET — Bootstrap 95% CIs ({args.n_boot} replicates)")
    print("=" * 72)
    test_raw = boot_df[boot_df["metric_set"] == "test_raw"].iloc[0]
    test_cal = boot_df[boot_df["metric_set"] == "test_calibrated"].iloc[0]
    for metric in ["roc_auc", "pr_auc", "f1", "brier_loss", "mcc"]:
        r_mean = test_raw[f"{metric}__mean"]
        r_lo = test_raw[f"{metric}__ci_lo"]
        r_hi = test_raw[f"{metric}__ci_hi"]
        c_mean = test_cal[f"{metric}__mean"]
        c_lo = test_cal[f"{metric}__ci_lo"]
        c_hi = test_cal[f"{metric}__ci_hi"]
        print(
            f"  {metric:14s}  raw: {r_mean:.4f} [{r_lo:.4f}, {r_hi:.4f}]   "
            f"cal: {c_mean:.4f} [{c_lo:.4f}, {c_hi:.4f}]"
        )
    print()
    print("CALIBRATION (ECE / MCE):")
    for _, row in ece_df.iterrows():
        print(f"  {row['eval_set']:18s}  ECE={row['ECE']:.4f}   MCE={row['MCE']:.4f}")
    print()
    print("OPERATING POINTS (test_calibrated):")
    for _, row in op_df[op_df["eval_set"] == "test_calibrated"].iterrows():
        status = "✓" if row["achieved"] else "✗"
        if row["achieved"]:
            print(
                f"  {status} {row['constraint']:18s}  threshold={row['threshold']:.3f}  "
                f"precision={row['precision']:.3f}  recall={row['recall']:.3f}"
            )
        else:
            print(f"  {status} {row['constraint']:18s}  not achievable")


if __name__ == "__main__":
    main()
