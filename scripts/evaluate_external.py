#!/usr/bin/env python3
"""Run external validation on all configured sources.

Usage
-----
    python scripts/evaluate_external.py                # all sources
    python scripts/evaluate_external.py --only denovo  # one source

The script performs the Phase-D pipeline end-to-end:

1. Load external source into a (variant_key, label, gene, …) table.
2. Featurize by joining on the cached dbNSFP parquet.
3. Refit the isotonic calibrator from the persisted val predictions.
4. Score, compute bootstrap 95% CIs, save CSV + parquet artifacts under
   `results/metrics/`.

Coverage is reported openly — rows without dbNSFP features are logged to
`results/metrics/external_<source>_unmapped.csv` so the external-validation
discussion can cite exact numbers instead of hiding them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.isotonic import IsotonicRegression
from src.external_validation.denovo_loader import load_denovo_db
from src.external_validation.evaluate import (
    ExternalEvalConfig,
    evaluate_external,
)
from src.external_validation.featurize import featurize_external
from src.external_validation.vep_featurize import (
    VEPFetchConfig,
    fetch_vep_features,
)
from src.gnomad_constraint import CONSTRAINT_COLS, load_constraint_table, merge_constraint

REPO = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "model": "results/checkpoints/xgboost_best.ubj",
    "feature_columns": "results/metrics/xgboost_feature_columns.csv",
    "train_split": "data/splits/train.parquet",
    "val_predictions": "results/metrics/xgboost_predictions.parquet",
    "dbnsfp_cache": "data/intermediate/dbnsfp_selected_features.parquet",
    "constraint_table": "data/raw/gnomad_constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz",
    "constraint_medians": "results/metrics/gnomad_constraint_medians.csv",
    "out_dir": "results/metrics",
    "figures_dir": "results/figures",
}


def _load_constraint_medians(path: Path) -> dict[str, float]:
    """Reload the train-fit gnomAD-constraint medians for imputing external rows."""
    df = pd.read_csv(path, index_col=0)
    return {k: float(v) for k, v in df["value"].items()}


def attach_gnomad_constraint(
    featurized: pd.DataFrame,
    *,
    constraint_path: Path,
    medians_path: Path,
) -> pd.DataFrame:
    """Merge pLI/LOEUF/mis_z/oe_mis_upper/lof_z onto the external feature frame.

    Uses train-fit medians (never refit from external data) so the external
    number stays comparable to the held-out test split. Adds
    `is_imputed_gnomad_constraint` flag to match the training schema.
    """
    if "gene" not in featurized.columns:
        raise ValueError("external table must have a `gene` column before constraint merge")
    # Drop any pre-existing constraint columns to avoid _x/_y pollution.
    overlap = [
        c for c in CONSTRAINT_COLS + ["is_imputed_gnomad_constraint"] if c in featurized.columns
    ]
    if overlap:
        featurized = featurized.drop(columns=overlap)
    constraint = load_constraint_table(constraint_path)
    medians = _load_constraint_medians(medians_path)
    merged, _ = merge_constraint(featurized, constraint=constraint, impute_medians=medians)
    return merged


SOURCES: dict[str, dict] = {
    "denovo_db": {
        "raw": "data/raw/external/denovo_db/denovo-db.non-ssc-samples.variants.tsv.gz",
        "loader": "denovo_db",
    },
}


def refit_isotonic(val_preds_path: Path) -> IsotonicRegression:
    """Refit isotonic regression on persisted validation predictions."""
    df = pd.read_parquet(val_preds_path)
    val = df[df["split"] == "val"]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val["p_raw"].to_numpy(), val["y_true"].to_numpy())
    return iso


def score_with_calibration(
    featurized: pd.DataFrame,
    *,
    calibrator: IsotonicRegression,
    config: ExternalEvalConfig,
    source_name: str,
    out_dir: Path,
):
    """Patch in the refit calibrator before calling evaluate_external."""
    # Monkey-attach calibrator to a temp file so evaluate() can load it.
    import os
    import tempfile

    import joblib

    tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
    joblib.dump(calibrator, tmp.name)
    tmp.close()
    try:
        config = ExternalEvalConfig(
            model_path=config.model_path,
            calibrator_path=Path(tmp.name),
            feature_columns_csv=config.feature_columns_csv,
            train_split_path=config.train_split_path,
            n_boot=config.n_boot,
            seed=config.seed,
        )
        return evaluate_external(
            featurized,
            config=config,
            source_name=source_name,
            out_dir=out_dir,
        )
    finally:
        os.unlink(tmp.name)


def run_source(name: str, cfg: dict, args) -> None:
    print(f"\n{'='*70}\nExternal source: {name}\n{'='*70}")
    raw = REPO / cfg["raw"]
    if not raw.exists():
        print(f"  [SKIP] raw file missing: {raw}")
        return

    # 1. Load
    if cfg["loader"] == "denovo_db":
        ext = load_denovo_db(raw)
    else:
        raise ValueError(f"unknown loader: {cfg['loader']}")
    print(
        f"  loaded {len(ext):,} missense variants "
        f"({int(ext['label'].sum()):,} pathogenic, "
        f"{int((ext['label']==0).sum()):,} control). "
        f"{ext.attrs.get('n_unmapped', 0)} unmapped at canonicalization."
    )

    # Optional subsampling for wall-clock budget.
    if args.sample:
        # Stratified sample — preserve label balance if possible.
        per = args.sample // 2
        parts = []
        for lbl in (0, 1):
            chunk = ext[ext["label"] == lbl]
            if len(chunk) > 0:
                parts.append(
                    chunk.sample(
                        min(per if lbl == 0 else args.sample - per, len(chunk)),
                        random_state=args.seed,
                    )
                )
        ext = pd.concat(parts).reset_index(drop=True)
        print(f"  subsampled to {len(ext):,} rows (stratified by label)")

    # 2. Featurize — try cache first, fall back to Ensembl VEP REST for
    #    any variants not in the cache.
    out_dir = REPO / DEFAULTS["out_dir"]
    feat = featurize_external(ext, dbnsfp_cache=REPO / DEFAULTS["dbnsfp_cache"])
    print("  cache-hit: " + feat.summary())
    if len(feat.unmapped) > 0 and args.use_vep:
        cache_dir = REPO / "data/intermediate/vep_external" / name
        print(f"  VEP REST fallback for {len(feat.unmapped):,} missing variants…")
        vep_rows = fetch_vep_features(
            feat.unmapped,
            cfg=VEPFetchConfig(cache_dir=cache_dir),
        )
        print(f"  VEP returned features for {len(vep_rows):,} rows")
        if len(vep_rows) > 0:
            vep_enriched = feat.unmapped.merge(vep_rows, on="variant_key", how="inner")
            # vep_enriched now has all needed feature cols + label/gene/etc.
            feat_rows = pd.concat([feat.featurized, vep_enriched], ignore_index=True, sort=False)
            still_missing = feat.unmapped[
                ~feat.unmapped["variant_key"].isin(vep_rows["variant_key"])
            ]
            feat = type(feat)(
                featurized=feat_rows,
                unmapped=still_missing,
                coverage=len(feat_rows) / max(len(ext), 1),
            )
            print("  after VEP: " + feat.summary())

    feat.unmapped.to_csv(out_dir / f"external_{name}_unmapped.csv", index=False)
    coverage_row = pd.DataFrame(
        [
            {
                "source": name,
                "n_raw": len(ext),
                "n_featurized": len(feat.featurized),
                "n_unmapped_dbnsfp": len(feat.unmapped),
                "coverage": feat.coverage,
            }
        ]
    )
    coverage_row.to_csv(out_dir / f"external_{name}_coverage.csv", index=False)

    # 2b. Attach gnomAD gene-level constraint (pLI / LOEUF / mis_z / ...)
    #     using train-fit medians so no leakage from the external set.
    constraint_path = REPO / DEFAULTS["constraint_table"]
    medians_path = REPO / DEFAULTS["constraint_medians"]
    if constraint_path.exists() and medians_path.exists() and len(feat.featurized) > 0:
        feat_with_constraint = attach_gnomad_constraint(
            feat.featurized,
            constraint_path=constraint_path,
            medians_path=medians_path,
        )
        imputed_rate = float(feat_with_constraint["is_imputed_gnomad_constraint"].mean())
        print(f"  gnomAD constraint merge: {1 - imputed_rate:.1%} gene-level coverage")
        feat = type(feat)(
            featurized=feat_with_constraint,
            unmapped=feat.unmapped,
            coverage=feat.coverage,
        )
    else:
        print("  [skip] gnomAD constraint merge (table or medians missing)")

    if len(feat.featurized) < 20:
        print(
            f"  [STOP] only {len(feat.featurized)} featurized rows — "
            f"too few for bootstrapped evaluation. Coverage CSV saved."
        )
        return

    # 3. Refit calibrator
    cal = refit_isotonic(REPO / DEFAULTS["val_predictions"])

    # 4. Evaluate
    cfg_obj = ExternalEvalConfig(
        model_path=REPO / DEFAULTS["model"],
        calibrator_path=None,  # patched by score_with_calibration
        feature_columns_csv=REPO / DEFAULTS["feature_columns"],
        train_split_path=REPO / DEFAULTS["train_split"],
        n_boot=args.n_boot,
        seed=args.seed,
    )
    result = score_with_calibration(
        feat.featurized,
        calibrator=cal,
        config=cfg_obj,
        source_name=name,
        out_dir=out_dir,
    )
    print(f"  saved metrics → {result['metrics_csv']}")
    for r in result["rows"]:
        if "note" in r:
            print(f"    [{r['slice']}] {r['note']}")
        else:
            print(
                f"    [{r['slice']}]  n={r['n']}  ROC={r['roc_auc']:.3f} "
                f"[{r['roc_auc_ci_lo']:.3f}, {r['roc_auc_ci_hi']:.3f}]  "
                f"PR={r['pr_auc']:.3f} "
                f"[{r['pr_auc_ci_lo']:.3f}, {r['pr_auc_ci_hi']:.3f}]"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(SOURCES) + ["all"], default="all")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Stratified subsample size (0 = all). Useful for VEP budget.",
    )
    ap.add_argument(
        "--use-vep",
        action="store_true",
        help="Fall back to Ensembl VEP REST for missing dbNSFP features",
    )
    args = ap.parse_args()

    targets = SOURCES if args.only == "all" else {args.only: SOURCES[args.only]}
    for name, cfg in targets.items():
        run_source(name, cfg, args)


if __name__ == "__main__":
    main()
