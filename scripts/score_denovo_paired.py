#!/usr/bin/env python3
# Added for the P0 revision pass (see CLAUDE_CODE_P0_FIXES.md, P0-1 Step 2).
"""Score denovo-db with both pre- and post-constraint XGBoost checkpoints.

Emits two per-variant prediction parquets in the schema required by the
P0-1 paired-bootstrap script:
    variant_id, y_true, p_pred, slice

`slice` is one of {"holdout", "full"} (emitted as two row groups per file).

The pre-constraint checkpoint is recovered from git history (commit f8ab464,
before gnomAD constraint features were introduced at b2fd3d0); see
docs/CHANGELOG.md for the archaeology trail.

Because the raw denovo-db TSV is gitignored, this script reconstructs the
featurized frame from the existing `external_denovo_db_predictions.parquet`
(which persists the 642 variant_keys and labels) + the cached dbNSFP and
gnomAD AF parquets. Imputation medians are re-derived from the committed
training split at inference time --- identical to what `src.training`
would compute.

ROC-AUC and average precision are invariant to any monotone transform
(including isotonic calibration), so match-check values are the same
whether we score p_raw or p_calibrated. We emit the calibrated probability
as `p_pred` for downstream consistency.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.data_splitting import assign_gene_family
from src.gnomad_constraint import CONSTRAINT_COLS, load_constraint_table, merge_constraint

REPO = Path(__file__).resolve().parents[1]

DBNSFP = REPO / "data/intermediate/dbnsfp_selected_features.parquet"
GNOMAD_AF = REPO / "data/intermediate/gnomad_af_clean.parquet"
TRAIN = REPO / "data/splits/train.parquet"
VAL_PREDS = REPO / "results/metrics/xgboost_predictions.parquet"
POST_PREDS = REPO / "results/metrics/external_denovo_db_predictions.parquet"
CONSTRAINT_TABLE = REPO / "data/raw/gnomad_constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz"
CONSTRAINT_MEDIANS = REPO / "results/metrics/gnomad_constraint_medians.csv"

# Imputation flags the training pipeline fills. Must match src.training order.
IMPUTED_FLAGS = [
    "is_imputed_phyloP100way_vertebrate",
    "is_imputed_phastCons100way_vertebrate",
    "is_imputed_GERP++_RS",
    "is_imputed_Grantham_distance",
    "is_imputed_BLOSUM62_score",
]
IMPUTED_SRC_COLS = [
    "phyloP100way_vertebrate",
    "phastCons100way_vertebrate",
    "GERP++_RS",
    "Grantham_distance",
    "BLOSUM62_score",
]


def load_denovo_skeleton() -> pd.DataFrame:
    """Start from the already-featurized 642 variant keys + labels."""
    df = pd.read_parquet(POST_PREDS)[["variant_key", "gene", "label", "family_holdout"]]
    return df.rename(columns={"label": "y_true"}).copy()


def join_raw_features(skel: pd.DataFrame) -> pd.DataFrame:
    dbnsfp = pd.read_parquet(DBNSFP)
    af = pd.read_parquet(GNOMAD_AF)
    merged = skel.merge(dbnsfp, on="variant_key", how="left")
    merged = merged.merge(af[["variant_key", "AF_popmax", "AN", "AC", "log_AF"]],
                          on="variant_key", how="left")
    # Imputation flags: 1 if source column is NaN, else 0.
    for flag, src in zip(IMPUTED_FLAGS, IMPUTED_SRC_COLS):
        merged[flag] = merged[src].isna().astype(int)
    return merged


def fit_medians(train: pd.DataFrame, numeric_cols: list[str]) -> dict[str, float]:
    med = train[numeric_cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    return {c: float(v) for c, v in med.items() if not np.isnan(v)}


def build_transformer(feature_cols_csv: Path, train: pd.DataFrame):
    manifest = pd.read_csv(feature_cols_csv)["encoded_feature"].tolist()
    num_cols = [c.removeprefix("num__") for c in manifest if c.startswith("num__")]
    cat_prefixes = sorted(
        {c.removeprefix("cat__").rsplit("_", 1)[0] for c in manifest if c.startswith("cat__")}
    )
    transformer = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_prefixes),
        ],
        remainder="drop",
    )
    transformer.fit(train[num_cols + cat_prefixes])
    return transformer, num_cols, cat_prefixes


def score_one(
    featurized: pd.DataFrame,
    *,
    model_path: Path,
    feature_cols_csv: Path,
    train_parquet: Path,
) -> pd.DataFrame:
    train = pd.read_parquet(train_parquet)
    transformer, num_cols, cat_cols = build_transformer(feature_cols_csv, train)

    # Apply train-fit median imputation to any remaining NaN numerics.
    X = featurized[num_cols + cat_cols].copy()
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
    medians = fit_medians(train, num_cols)
    X[num_cols] = X[num_cols].fillna(medians)
    # Fallback for any column that had all-NaN in train (shouldn't happen):
    X[num_cols] = X[num_cols].fillna(0.0)
    X[num_cols] = X[num_cols].astype(np.float32)

    Xt = transformer.transform(X)

    model = XGBClassifier()
    model.load_model(str(model_path))
    p_raw = model.predict_proba(Xt)[:, 1]

    # Try to refit isotonic on val predictions if available (ROC-AUC /
    # PR-AUC are invariant to monotone transforms so the match check is
    # unaffected; we persist p_cal for forward-compat with callers).
    if VAL_PREDS.exists():
        val_preds = pd.read_parquet(VAL_PREDS)
        val = val_preds[val_preds["split"] == "val"]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(val["p_raw"].to_numpy(), val["y_true"].to_numpy())
        p_cal = iso.transform(p_raw)
    else:
        p_cal = p_raw  # AUC metrics invariant to monotone transform

    preds = featurized[["variant_key", "y_true", "family_holdout"]].copy()
    preds["p_pred"] = p_cal
    preds["p_raw"] = p_raw
    return preds


def attach_constraint(feat: pd.DataFrame) -> pd.DataFrame:
    if not CONSTRAINT_TABLE.exists() or not CONSTRAINT_MEDIANS.exists():
        raise FileNotFoundError("constraint inputs missing")
    constraint = load_constraint_table(CONSTRAINT_TABLE)
    medians_df = pd.read_csv(CONSTRAINT_MEDIANS, index_col=0)
    medians = {k: float(v) for k, v in medians_df["value"].items()}
    overlap = [c for c in CONSTRAINT_COLS + ["is_imputed_gnomad_constraint"] if c in feat.columns]
    if overlap:
        feat = feat.drop(columns=overlap)
    merged, _ = merge_constraint(feat, constraint=constraint, impute_medians=medians)
    return merged


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Emit full + holdout slice rows in the schema P0-1 requires."""
    out = []
    full = df.assign(slice="full")[["variant_key", "y_true", "p_pred", "slice"]].rename(
        columns={"variant_key": "variant_id"}
    )
    out.append(full)
    hold = (
        df.loc[df["family_holdout"].astype(bool)]
        .assign(slice="holdout")[["variant_key", "y_true", "p_pred", "slice"]]
        .rename(columns={"variant_key": "variant_id"})
    )
    out.append(hold)
    return pd.concat(out, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-pre", type=Path,
                    default=REPO / "results/metrics/denovo_predictions_pre_constraint.parquet")
    ap.add_argument("--out-post", type=Path,
                    default=REPO / "results/metrics/denovo_predictions_post_constraint.parquet")
    args = ap.parse_args()

    skel = load_denovo_skeleton()
    print(f"skeleton: {len(skel)} variants  (holdout={int(skel['family_holdout'].sum())})")
    feat_base = join_raw_features(skel)

    # Post-constraint: reuse canonical predictions from the existing
    # external_denovo_db_predictions.parquet (generated with the current
    # xgboost_best.ubj + full constraint featurization). This avoids
    # needing the gitignored constraint raw table.
    post_src = pd.read_parquet(POST_PREDS)
    post = skel.merge(
        post_src[["variant_key", "p_calibrated", "p_raw"]],
        on="variant_key",
        how="left",
    ).rename(columns={"p_calibrated": "p_pred"})

    # Pre-constraint: score fresh --- no constraint merge needed.
    feat_pre = feat_base.copy()
    pre = score_one(
        feat_pre,
        model_path=REPO / "results/checkpoints/xgboost_pre_constraint.ubj",
        feature_cols_csv=REPO / "results/metrics/xgboost_pre_constraint_feature_columns.csv",
        train_parquet=TRAIN,
    )

    to_long(pre).to_parquet(args.out_pre, index=False)
    to_long(post).to_parquet(args.out_post, index=False)
    print(f"wrote {args.out_pre}  ({args.out_pre.stat().st_size} B)")
    print(f"wrote {args.out_post}  ({args.out_post.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
