#!/usr/bin/env python3
# Added for Phase 2.1 (S8): score Phase-2.1 XGBoost on denovo-db.
"""Score the Phase-2.1 XGBoost model on the denovo-db external cohort.

Reconstructs the featurized denovo-db frame from:
  - the existing `external_denovo_db_predictions.parquet` (skeleton:
    variant_key, gene, label, family_holdout)
  - dbNSFP cache + gnomAD-AF cache (committed intermediates)
  - gnomAD constraint medians (committed CSV; raw constraint table is
    not in the worktree, so all denovo rows are flagged
    is_imputed_gnomad_constraint=1, matching the Phase-1 behavior on
    long-tail genes)
  - esm2_denovo_db_scores.parquet (esm2_llr column, aggregated by
    min per variant_key)

Then trains the Phase-2.1 isotonic calibrator on Phase-2.1 val
predictions and applies it to the denovo-db scores. Outputs
`external_denovo_db_predictions_phase21.parquet` and
`external_denovo_db_metrics_phase21.csv` in the schema used by
the Phase-1 equivalents.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

REPO = Path(__file__).resolve().parents[1]
SKELETON = REPO / "results/metrics/external_denovo_db_predictions.parquet"
DBNSFP = REPO / "data/intermediate/dbnsfp_selected_features.parquet"
GNOMAD_AF = REPO / "data/intermediate/gnomad_af_clean.parquet"
CONSTRAINT_MEDIANS = REPO / "results/metrics/gnomad_constraint_medians.csv"
ESM2_DENOVO = REPO / "results/metrics/esm2_denovo_db_scores.parquet"
PHASE21_VAL = REPO / "data/splits/phase21/val.parquet"
PHASE21_TRAIN = REPO / "data/splits/phase21/train.parquet"
PHASE21_TEST = REPO / "data/splits/phase21/test.parquet"
PHASE21_FEATCOLS = REPO / "results/metrics/xgboost_phase21_feature_columns.csv"
PHASE21_MODEL = REPO / "results/checkpoints/xgboost_phase21_esm2.ubj"

OUT_PRED = REPO / "results/metrics/external_denovo_db_predictions_phase21.parquet"
OUT_METRICS = REPO / "results/metrics/external_denovo_db_metrics_phase21.csv"
OUT_COVERAGE = REPO / "results/metrics/external_denovo_db_coverage_phase21.csv"

CONSTRAINT_COLS = ["pLI", "oe_lof_upper", "mis_z", "oe_mis_upper", "lof_z"]
IMPUTED_FLAGS = [
    "is_imputed_phyloP100way_vertebrate",
    "is_imputed_phastCons100way_vertebrate",
    "is_imputed_GERP++_RS",
    "is_imputed_Grantham_distance",
    "is_imputed_BLOSUM62_score",
]
IMPUTED_SRC = [
    "phyloP100way_vertebrate",
    "phastCons100way_vertebrate",
    "GERP++_RS",
    "Grantham_distance",
    "BLOSUM62_score",
]


def main() -> int:
    skel = pd.read_parquet(SKELETON)[["variant_key", "gene", "label", "family_holdout"]]
    print(f"[denovo_phase21] skeleton {len(skel)} variants ({int(skel['family_holdout'].sum())} holdout)")

    dbnsfp = pd.read_parquet(DBNSFP)
    af = pd.read_parquet(GNOMAD_AF)
    constraint_medians = pd.read_csv(CONSTRAINT_MEDIANS, index_col=0)["value"].astype(float).to_dict()
    esm2 = pd.read_parquet(ESM2_DENOVO)[["variant_key", "esm2_llr"]].dropna(subset=["esm2_llr"])
    esm2 = esm2.groupby("variant_key", as_index=False)["esm2_llr"].min()

    feat = skel.merge(dbnsfp, on="variant_key", how="left")
    feat = feat.merge(af[["variant_key", "AF_popmax", "AN", "AC", "log_AF"]], on="variant_key", how="left")
    for flag, src in zip(IMPUTED_FLAGS, IMPUTED_SRC):
        feat[flag] = feat[src].isna().astype(int)

    # Constraint: rebuild gene -> constraint lookup from train.parquet (which has
    # the per-row constraint values produced by the original Phase-1 pipeline).
    # Fall back to medians for any denovo gene NOT in train. Mirrors what the
    # canonical evaluate_external would do given the raw table; required because
    # the raw constraint table is gitignored in this worktree.
    train = pd.read_parquet(PHASE21_TRAIN)
    gene_constraint = (
        train.dropna(subset=CONSTRAINT_COLS, how="all")
        .groupby("gene")[CONSTRAINT_COLS]
        .first()
        .reset_index()
    )
    feat = feat.merge(gene_constraint, on="gene", how="left")
    is_imputed_constraint = feat[CONSTRAINT_COLS[0]].isna().astype(int)
    for c in CONSTRAINT_COLS:
        feat[c] = feat[c].fillna(constraint_medians[c])
    feat["is_imputed_gnomad_constraint"] = is_imputed_constraint
    print(
        f"[denovo_phase21] constraint coverage: "
        f"{(1 - feat['is_imputed_gnomad_constraint'].mean()):.4f} "
        f"({(feat['is_imputed_gnomad_constraint']==0).sum()}/{len(feat)} have real values)"
    )

    feat = feat.merge(esm2, on="variant_key", how="left")
    feat["is_imputed_esm2_llr"] = feat["esm2_llr"].isna().astype(int)
    print(f"[denovo_phase21] esm2 coverage: {feat['esm2_llr'].notna().mean():.4f}")

    # Build transformer from Phase-2.1 train; impute numerics with train medians.
    train = pd.read_parquet(PHASE21_TRAIN)
    manifest = pd.read_csv(PHASE21_FEATCOLS)["encoded_feature"].tolist()
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

    X = feat[num_cols + cat_prefixes].copy()
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
    medians = train[num_cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    X[num_cols] = X[num_cols].fillna(medians).fillna(0.0).astype(np.float32)
    Xt = transformer.transform(X)

    booster = xgb.Booster()
    booster.load_model(str(PHASE21_MODEL))
    p_raw = booster.predict(xgb.DMatrix(Xt))

    # Refit isotonic on Phase-2.1 val raw probs
    val = pd.read_parquet(PHASE21_VAL)
    val_X = val[num_cols + cat_prefixes].copy()
    val_X[num_cols] = val_X[num_cols].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0).astype(np.float32)
    val_Xt = transformer.transform(val_X)
    val_p_raw = booster.predict(xgb.DMatrix(val_Xt))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val_p_raw, val["label"].astype(int))
    p_cal = iso.transform(p_raw)

    out = feat[["variant_key", "gene", "label", "family_holdout"]].copy()
    out["p_raw"] = p_raw
    out["p_calibrated"] = p_cal
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PRED, index=False)
    print(f"[denovo_phase21] wrote {OUT_PRED}")

    # Metrics on full + family-holdout slices
    rows = []
    for slice_name, mask in (
        ("full", np.ones(len(out), dtype=bool)),
        ("family_holdout_only", out["family_holdout"].astype(bool).to_numpy()),
    ):
        sub = out[mask]
        y = sub["label"].astype(int).to_numpy()
        if y.sum() in (0, len(y)):
            rows.append({"slice": slice_name, "n": int(mask.sum()), "note": "single-class"})
            continue
        roc = float(roc_auc_score(y, sub["p_calibrated"]))
        pr = float(average_precision_score(y, sub["p_calibrated"]))
        rows.append({
            "slice": slice_name,
            "n": int(mask.sum()),
            "n_positive": int(y.sum()),
            "roc_auc_calibrated": roc,
            "pr_auc_calibrated": pr,
            "roc_auc_raw": float(roc_auc_score(y, sub["p_raw"])),
            "pr_auc_raw": float(average_precision_score(y, sub["p_raw"])),
        })
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_METRICS, index=False)
    print(f"[denovo_phase21] wrote {OUT_METRICS}")
    print(metrics_df.to_string(index=False))

    cov = pd.DataFrame([{
        "source": "denovo_db",
        "n_total": len(out),
        "n_holdout": int(out["family_holdout"].sum()),
        "esm2_coverage": float(feat["esm2_llr"].notna().mean()),
        "constraint_imputed_pct": 1.0,
    }])
    cov.to_csv(OUT_COVERAGE, index=False)
    print(f"[denovo_phase21] wrote {OUT_COVERAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
