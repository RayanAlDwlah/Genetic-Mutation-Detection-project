#!/usr/bin/env python3
# Added for Phase 2.1 (S3/S4): train XGBoost with esm2_llr as a feature.
"""Train Phase-2.1 XGBoost using either frozen Phase-1 hyperparameters
(headline path) or full Optuna retune (appendix sensitivity).

Reuses src.training utilities (select_feature_columns, prepare_split_features)
so the only thing that changes vs Phase-1 is the input split parquets and
the output namespace. Phase-1 artifacts are NEVER overwritten.

Headline rule (per S4 of plan): the frozen-hp model is the canonical
result; Optuna retune outputs are written to a parallel namespace and
are appendix-only.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import compute_classification_metrics, select_best_threshold
from src.models.xgboost_model import (
    XGBTuningConfig,
    build_xgboost_model,
    tune_xgboost,
)
from src.training import prepare_split_features, select_feature_columns
from src.utils import require_file, resolve_path

REPO = Path(__file__).resolve().parents[1]

PHASE1_PARAMS_KEYS = [
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "colsample_bylevel",
    "gamma",
    "reg_alpha",
    "reg_lambda",
    "max_delta_step",
    "scale_pos_weight",
]


def _coerce(value: float, key: str) -> float | int:
    """XGBoost wants ints for tree-shape params, floats for everything else."""
    int_keys = {"max_depth", "min_child_weight", "max_delta_step"}
    return int(value) if key in int_keys else float(value)


def load_phase1_params(params_csv: Path) -> dict[str, float | int]:
    df = pd.read_csv(params_csv)
    if df.empty:
        raise ValueError(f"empty params CSV: {params_csv}")
    row = df.iloc[0]
    out: dict[str, float | int] = {}
    for k in PHASE1_PARAMS_KEYS:
        if k in row.index and pd.notna(row[k]):
            out[k] = _coerce(row[k], k)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/splits/phase21/train.parquet")
    p.add_argument("--val", default="data/splits/phase21/val.parquet")
    p.add_argument("--test", default="data/splits/phase21/test.parquet")
    p.add_argument(
        "--frozen-params",
        default=None,
        help="Path to phase-1 best_params CSV; if set, skip Optuna and use these",
    )
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prefix",
        default="phase21",
        help="Output filename prefix; e.g. 'phase21' or 'phase21_optuna'",
    )
    p.add_argument(
        "--checkpoints-dir",
        default="results/checkpoints",
        help="Where to write the .ubj model file",
    )
    p.add_argument(
        "--metrics-dir",
        default="results/metrics",
        help="Where to write CSV outputs",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_path = resolve_path(REPO, args.train)
    val_path = resolve_path(REPO, args.val)
    test_path = resolve_path(REPO, args.test)
    for p in (train_path, val_path, test_path):
        require_file(p)

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    numeric_cols, categorical_cols = select_feature_columns(train_df)
    print(f"[train_phase21] selected {len(numeric_cols)} numeric + {len(categorical_cols)} categorical")
    if "esm2_llr" not in numeric_cols:
        raise SystemExit(
            "ABORT: esm2_llr is not in the numeric feature list. "
            "Did you build phase21 splits with scripts/build_phase21_train.py?"
        )

    x_train, x_val, x_test, encoded_feature_names = prepare_split_features(
        train_df, val_df, test_df, numeric_cols, categorical_cols
    )
    y_train = train_df["label"].astype(int)
    y_val = val_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = float(neg / max(pos, 1))

    if args.frozen_params:
        # Headline path: skip Optuna; train one model with phase-1 params
        params_csv = resolve_path(REPO, args.frozen_params)
        require_file(params_csv)
        frozen = load_phase1_params(params_csv)
        # Phase-1 scale_pos_weight may be slightly different on phase21
        # splits (same rows), but recompute to keep the loss function
        # ratio-matched to the actual training data.
        frozen["scale_pos_weight"] = scale_pos_weight
        print(f"[train_phase21] frozen-hp mode; params from {params_csv}")
        print(f"  {frozen}")
        best_model = build_xgboost_model(
            frozen,
            seed=args.seed,
            n_estimators=2500,
            early_stopping_rounds=80,
        )
        best_model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        history = pd.DataFrame(
            [
                {
                    "trial": 0,
                    "score": float("nan"),
                    "val_roc_auc": float("nan"),
                    "val_pr_auc": float("nan"),
                    **frozen,
                    "frozen": True,
                }
            ]
        )
        best_params = frozen
    else:
        print(f"[train_phase21] full-Optuna mode; n_trials={args.trials}")
        cfg = XGBTuningConfig(
            n_trials=max(2, int(args.trials)),
            seed=int(args.seed),
            n_estimators=2500,
            early_stopping_rounds=80,
        )
        best_model, best_params, history = tune_xgboost(
            x_train,
            y_train,
            x_val,
            y_val,
            config=cfg,
            scale_pos_weight=scale_pos_weight,
        )

    val_prob = best_model.predict_proba(x_val)[:, 1]
    best_threshold, threshold_curve = select_best_threshold(y_val, val_prob)

    rows = []
    for split_name, x, y in (
        ("train", x_train, y_train),
        ("val", x_val, y_val),
        ("test", x_test, y_test),
    ):
        prob = best_model.predict_proba(x)[:, 1]
        row = compute_classification_metrics(y, prob, threshold=best_threshold)
        row["split"] = split_name
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df[["split"] + [c for c in metrics_df.columns if c != "split"]]

    ckpt = REPO / args.checkpoints_dir / f"xgboost_{args.prefix}_esm2.ubj"
    metrics_dir = REPO / args.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    best_model.save_model(str(ckpt))
    history.to_csv(metrics_dir / f"xgboost_{args.prefix}_tuning_history.csv", index=False)
    threshold_curve.to_csv(metrics_dir / f"xgboost_{args.prefix}_val_threshold_curve.csv", index=False)
    metrics_df.to_csv(metrics_dir / f"xgboost_{args.prefix}_split_metrics.csv", index=False)
    pd.DataFrame({"encoded_feature": encoded_feature_names}).to_csv(
        metrics_dir / f"xgboost_{args.prefix}_feature_columns.csv", index=False
    )
    selected = (
        history.iloc[0].get("score", float("nan")) if not history.empty else float("nan")
    )
    pd.DataFrame(
        [
            {
                **best_params,
                "best_threshold": float(best_threshold),
                "selected_trial_score": float(selected) if pd.notna(selected) else float("nan"),
                "n_numeric_features": len(numeric_cols),
                "n_categorical_features": len(categorical_cols),
                "n_encoded_features": len(encoded_feature_names),
                "categorical_columns": "|".join(categorical_cols),
                "frozen_hp": bool(args.frozen_params),
            }
        ]
    ).to_csv(metrics_dir / f"xgboost_{args.prefix}_best_params.csv", index=False)

    print(f"[train_phase21] saved model -> {ckpt}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
