#!/usr/bin/env python3
"""Merge gnomAD v2.1.1 per-gene constraint metrics into the variant splits.

Context
-------
Phase D (denovo-db) revealed that the tabular baseline performs at chance
on affected-vs-control de-novo missense. Diagnosis: the model sees
*variant-level* signals (conservation, AA chemistry) but never the
*gene-level* constraint prior that clinicians actually use — "this gene is
intolerant to any damage" vs "this gene is a passenger."

gnomAD publishes per-gene constraint summaries that encode exactly this:
- `pLI`            : probability of LoF intolerance (0–1)
- `oe_lof_upper`   : LOEUF — upper CI of observed/expected LoF (lower = more constrained)
- `mis_z`          : missense Z-score (higher = more constrained)
- `oe_mis_upper`   : missense observed/expected upper CI
- `lof_z`          : LoF Z-score

This module joins these onto any (train/val/test/external) parquet by gene.
Missing genes (≈7.5% of the training cohort) get median-imputed values plus
`is_imputed_gnomad_constraint` = 1 so SHAP can separate imputed rows.

Run:
    python -m src.gnomad_constraint

which rewrites `data/splits/{train,val,test}.parquet` in place with the
new columns appended.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

CONSTRAINT_FILE = "data/raw/gnomad_constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz"

CONSTRAINT_COLS = ["pLI", "oe_lof_upper", "mis_z", "oe_mis_upper", "lof_z"]

# Human-readable aliases published in the downstream feature manifest.
COLUMN_DOC = {
    "pLI": "Probability of LoF intolerance (Lek 2016). 0=tolerant, 1=intolerant.",
    "oe_lof_upper": "LOEUF — upper CI of observed/expected LoF. Lower = more constrained.",
    "mis_z": "Missense Z-score. Higher = gene tolerates less missense variation.",
    "oe_mis_upper": "Upper CI of observed/expected missense. Lower = more constrained.",
    "lof_z": "LoF Z-score. Higher = gene tolerates fewer LoF variants.",
}


def load_constraint_table(path: Path = REPO / CONSTRAINT_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    keep = ["gene"] + [c for c in CONSTRAINT_COLS if c in df.columns]
    df = df[keep].copy()
    # gnomAD has one row per canonical transcript; some genes have duplicates
    # across assemblies — keep the row with lowest oe_lof_upper (most
    # constrained), falling back to first.
    if df["gene"].duplicated().any():
        df = (
            df.sort_values("oe_lof_upper", na_position="last")
            .drop_duplicates("gene", keep="first")
            .reset_index(drop=True)
        )
    return df


def merge_constraint(
    variants: pd.DataFrame,
    *,
    constraint: pd.DataFrame,
    impute_medians: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Left-join constraint onto variant table by gene; median-impute misses.

    Returns (merged_frame, medians_used). Pass `impute_medians` from the train
    merge into the val/test/external merges to avoid leakage.
    """
    if "gene" not in variants.columns:
        raise ValueError("variant table must have a `gene` column")
    merged = variants.merge(constraint, on="gene", how="left", indicator=True)

    if impute_medians is None:
        # Fit medians from the rows where the gene was present (train-only use).
        impute_medians = {
            c: float(merged.loc[merged["_merge"] == "both", c].median())
            for c in CONSTRAINT_COLS
            if c in merged.columns
        }

    for c in CONSTRAINT_COLS:
        if c not in merged.columns:
            continue
        merged[c] = merged[c].fillna(impute_medians[c])

    merged["is_imputed_gnomad_constraint"] = (merged["_merge"] != "both").astype(int)
    merged = merged.drop(columns="_merge")
    return merged, impute_medians


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/splits/train.parquet")
    ap.add_argument("--val", default="data/splits/val.parquet")
    ap.add_argument("--test", default="data/splits/test.parquet")
    ap.add_argument("--constraint", default=CONSTRAINT_FILE)
    args = ap.parse_args()

    constraint = load_constraint_table(REPO / args.constraint)
    print(
        f"loaded {len(constraint):,} gene constraint rows "
        f"({constraint['pLI'].notna().sum():,} with pLI)"
    )

    train = pd.read_parquet(REPO / args.train)
    val = pd.read_parquet(REPO / args.val)
    test = pd.read_parquet(REPO / args.test)

    # 1. Fit medians on TRAIN only (avoid leakage).
    train_m, medians = merge_constraint(train, constraint=constraint)
    val_m, _ = merge_constraint(val, constraint=constraint, impute_medians=medians)
    test_m, _ = merge_constraint(test, constraint=constraint, impute_medians=medians)

    print("\nImpute medians (train-fit):")
    for k, v in medians.items():
        print(f"  {k}: {v:.4g}")

    coverage = {
        "train": 1 - train_m["is_imputed_gnomad_constraint"].mean(),
        "val": 1 - val_m["is_imputed_gnomad_constraint"].mean(),
        "test": 1 - test_m["is_imputed_gnomad_constraint"].mean(),
    }
    print("\nConstraint coverage (1 - imputed rate):")
    for k, v in coverage.items():
        print(f"  {k}: {v:.1%}")

    # Write back.
    train_m.to_parquet(REPO / args.train, index=False)
    val_m.to_parquet(REPO / args.val, index=False)
    test_m.to_parquet(REPO / args.test, index=False)
    print(f"\nRewrote splits with {len(CONSTRAINT_COLS) + 1} new columns:")
    for c in CONSTRAINT_COLS + ["is_imputed_gnomad_constraint"]:
        print(f"  {c}")

    # Persist medians for reuse by external-validation featurizer.
    pd.Series(medians).to_csv(
        REPO / "results/metrics/gnomad_constraint_medians.csv", header=["value"]
    )


if __name__ == "__main__":
    main()
