#!/usr/bin/env python3
"""Merge ClinVar labels + dbNSFP features + gnomAD frequencies.

Merge order (Option A):
1) ClinVar INNER JOIN dbNSFP on variant_key
   - Acts as missense filter, because dbNSFP is missense-focused.
2) LEFT JOIN gnomAD on variant_key
   - Unmatched rows get AF defaults (ultra-rare proxy).
"""

from __future__ import annotations

from math import log10
from pathlib import Path

import pandas as pd
from output import echo


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = REPO_ROOT / "data" / "intermediate"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

CLINVAR_PATH = INTERMEDIATE_DIR / "clinvar_labeled_clean.parquet"
GNOMAD_PATH = INTERMEDIATE_DIR / "gnomad_af_clean.parquet"
DBNSFP_PATH = INTERMEDIATE_DIR / "dbnsfp_selected_features.parquet"

OUTPUT_PATH = PROCESSED_DIR / "merged_clinvar_gnomad_dbnsfp.parquet"

GNOMAD_DEFAULT_AF = 0.0
GNOMAD_DEFAULT_AF_POPMAX = 0.0
GNOMAD_DEFAULT_LOG_AF = log10(1e-8)
GNOMAD_DEFAULT_IS_COMMON = False


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def _require_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _drop_duplicate_variant_keys(df: pd.DataFrame, table_name: str) -> tuple[pd.DataFrame, int]:
    dup_count = int(df.duplicated(subset=["variant_key"]).sum())
    if dup_count > 0:
        echo(f"{table_name}: found {dup_count:,} duplicate rows by variant_key; keeping first.")
        df = df.drop_duplicates(subset=["variant_key"], keep="first").copy()
    return df, dup_count


def _run_sanity_checks(merged: pd.DataFrame, expected_rows: int) -> None:
    if len(merged) != expected_rows:
        raise AssertionError(
            f"Row count changed after gnomAD LEFT join: expected {expected_rows:,}, got {len(merged):,}"
        )

    duplicate_keys = int(merged.duplicated(subset=["variant_key"]).sum())
    if duplicate_keys > 0:
        raise AssertionError(f"Found duplicate variant_keys after merge: {duplicate_keys:,}")

    label_nan = int(merged["label"].isna().sum())
    if label_nan > 0:
        raise AssertionError(f"Found NaN labels after merge: {label_nan:,}")

    gene_nan = int(merged["gene"].isna().sum())
    if gene_nan > 0:
        raise AssertionError(f"Found NaN genes after merge: {gene_nan:,}")


def main() -> None:
    _require_file(CLINVAR_PATH)
    _require_file(DBNSFP_PATH)
    _require_file(GNOMAD_PATH)

    echo(f"Loading ClinVar: {CLINVAR_PATH}")
    clinvar = pd.read_parquet(CLINVAR_PATH)
    _require_columns(
        clinvar,
        ["variant_key", "gene", "label", "review_stars", "chr", "pos", "ref", "alt"],
        "ClinVar",
    )
    clinvar, clinvar_dup = _drop_duplicate_variant_keys(clinvar, "ClinVar")
    clinvar_rows = len(clinvar)
    echo(f"ClinVar rows: {clinvar_rows:,}")
    if clinvar_dup:
        echo(f"ClinVar duplicate keys removed: {clinvar_dup:,}")

    echo(f"Loading dbNSFP: {DBNSFP_PATH}")
    dbnsfp = pd.read_parquet(DBNSFP_PATH)
    _require_columns(dbnsfp, ["variant_key"], "dbNSFP")

    # Keep dbNSFP columns except keys that would collide with ClinVar core columns.
    drop_collisions = {"chr", "pos", "ref", "alt", "gene", "label", "review_stars"}
    dbnsfp_feature_cols = [c for c in dbnsfp.columns if c not in drop_collisions]
    if "variant_key" not in dbnsfp_feature_cols:
        dbnsfp_feature_cols = ["variant_key", *dbnsfp_feature_cols]

    dbnsfp = dbnsfp[dbnsfp_feature_cols].copy()
    dbnsfp, dbnsfp_dup = _drop_duplicate_variant_keys(dbnsfp, "dbNSFP")

    # Option A fix: INNER JOIN to apply missense filtering via dbNSFP coverage.
    merged = clinvar.merge(dbnsfp, on="variant_key", how="inner", validate="one_to_one")
    rows_after_dbnsfp_inner = len(merged)
    rows_filtered_out = clinvar_rows - rows_after_dbnsfp_inner

    approx_k = f"~{round(rows_after_dbnsfp_inner / 1000):,}K"
    echo(
        f"Missense filter applied via dbNSFP inner join: "
        f"{clinvar_rows:,} -> {approx_k} rows"
    )
    echo(
        f"After dbNSFP inner join: {rows_after_dbnsfp_inner:,} rows "
        f"(filtered out {rows_filtered_out:,} from ClinVar)"
    )
    if dbnsfp_dup:
        echo(f"dbNSFP duplicate keys removed before merge: {dbnsfp_dup:,}")

    echo(f"Loading gnomAD: {GNOMAD_PATH}")
    gnomad = pd.read_parquet(GNOMAD_PATH)
    _require_columns(gnomad, ["variant_key", "AF", "AF_popmax", "log_AF", "is_common"], "gnomAD")

    gnomad_cols = [
        c
        for c in ["variant_key", "AF", "AF_popmax", "AN", "AC", "log_AF", "is_common"]
        if c in gnomad.columns
    ]
    gnomad = gnomad[gnomad_cols].copy()
    gnomad, gnomad_dup = _drop_duplicate_variant_keys(gnomad, "gnomAD")
    gnomad_keys = set(gnomad["variant_key"].astype(str))

    # Keep all missense-filtered rows; enrich with gnomAD when available.
    merged = merged.merge(gnomad, on="variant_key", how="left", validate="one_to_one")

    gnomad_matched = int(merged["variant_key"].isin(gnomad_keys).sum())
    gnomad_unmatched = rows_after_dbnsfp_inner - gnomad_matched

    gnomad_missing_mask = ~merged["variant_key"].isin(gnomad_keys)
    merged.loc[gnomad_missing_mask, "AF"] = GNOMAD_DEFAULT_AF
    merged.loc[gnomad_missing_mask, "AF_popmax"] = GNOMAD_DEFAULT_AF_POPMAX
    merged.loc[gnomad_missing_mask, "log_AF"] = GNOMAD_DEFAULT_LOG_AF
    merged.loc[gnomad_missing_mask, "is_common"] = GNOMAD_DEFAULT_IS_COMMON
    if "AN" in merged.columns:
        merged.loc[gnomad_missing_mask, "AN"] = merged.loc[gnomad_missing_mask, "AN"].fillna(0)
    if "AC" in merged.columns:
        merged.loc[gnomad_missing_mask, "AC"] = merged.loc[gnomad_missing_mask, "AC"].fillna(0)

    merged["is_common"] = merged["is_common"].fillna(False).astype(bool)
    merged["has_dbnsfp_features"] = True  # Always true after INNER join with dbNSFP.

    echo(
        f"After gnomAD merge: matched={gnomad_matched:,}, "
        f"unmatched={gnomad_unmatched:,} (AF defaults applied)"
    )
    if gnomad_dup:
        echo(f"gnomAD duplicate keys removed before merge: {gnomad_dup:,}")

    _run_sanity_checks(merged, expected_rows=rows_after_dbnsfp_inner)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)

    echo(f"Final merged dataset: {len(merged):,} rows, {len(merged.columns):,} columns")
    echo(f"Saved merged parquet: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
