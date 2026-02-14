#!/usr/bin/env python3
"""Merge ClinVar labels + gnomAD frequencies + dbNSFP features.

Purpose:
- Build one unified dataset for downstream modeling.
- Keep ClinVar as the primary table (LEFT table).
- Add gnomAD frequencies and dbNSFP features on top of ClinVar variants.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from math import log10
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = REPO_ROOT / "data" / "intermediate"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

CLINVAR_PATH = INTERMEDIATE_DIR / "clinvar_labeled_clean.parquet"
GNOMAD_PATH = INTERMEDIATE_DIR / "gnomad_af_clean.parquet"
DBNSFP_PATH = INTERMEDIATE_DIR / "dbnsfp_selected_features.parquet"

OUTPUT_PATH = PROCESSED_DIR / "merged_clinvar_gnomad_dbnsfp.parquet"
METADATA_PATH = PROCESSED_DIR / "merge_metadata.json"

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
        print(f"{table_name}: found {dup_count:,} duplicate rows by variant_key; keeping first.")
        df = df.drop_duplicates(subset=["variant_key"], keep="first").copy()
    return df, dup_count


def _run_sanity_checks(merged: pd.DataFrame, clinvar_rows: int) -> None:
    if len(merged) != clinvar_rows:
        raise AssertionError(
            f"Row count changed after LEFT joins: expected {clinvar_rows:,}, got {len(merged):,}"
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
    _require_file(GNOMAD_PATH)
    _require_file(DBNSFP_PATH)

    print(f"Loading ClinVar: {CLINVAR_PATH}")
    clinvar = pd.read_parquet(CLINVAR_PATH)
    _require_columns(
        clinvar,
        ["variant_key", "gene", "label", "review_stars", "chr", "pos", "ref", "alt"],
        "ClinVar",
    )
    clinvar, clinvar_dup = _drop_duplicate_variant_keys(clinvar, "ClinVar")
    clinvar_rows = len(clinvar)
    print(f"ClinVar rows: {clinvar_rows:,}")
    if clinvar_dup:
        print(f"ClinVar duplicate keys removed: {clinvar_dup:,}")

    print(f"Loading gnomAD: {GNOMAD_PATH}")
    gnomad = pd.read_parquet(GNOMAD_PATH)
    _require_columns(gnomad, ["variant_key", "AF", "AF_popmax", "log_AF", "is_common"], "gnomAD")
    gnomad_cols = [c for c in ["variant_key", "AF", "AF_popmax", "AN", "AC", "log_AF", "is_common"] if c in gnomad.columns]
    gnomad = gnomad[gnomad_cols].copy()
    gnomad, gnomad_dup = _drop_duplicate_variant_keys(gnomad, "gnomAD")
    gnomad_keys = set(gnomad["variant_key"].astype(str))

    merged = clinvar.merge(gnomad, on="variant_key", how="left", validate="one_to_one")
    gnomad_matched = int(merged["variant_key"].isin(gnomad_keys).sum())
    gnomad_unmatched = clinvar_rows - gnomad_matched

    # Variants absent from gnomAD are treated as ultra-rare by default.
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
    print(f"After gnomAD merge: matched={gnomad_matched:,}, unmatched={gnomad_unmatched:,} (AF defaults applied)")
    if gnomad_dup:
        print(f"gnomAD duplicate keys removed before merge: {gnomad_dup:,}")

    print(f"Loading dbNSFP: {DBNSFP_PATH}")
    dbnsfp = pd.read_parquet(DBNSFP_PATH)
    _require_columns(dbnsfp, ["variant_key"], "dbNSFP")

    drop_collisions = {"chr", "pos", "ref", "alt", "gene", "label", "review_stars"}
    dbnsfp_feature_cols = [c for c in dbnsfp.columns if c not in drop_collisions]
    if "variant_key" not in dbnsfp_feature_cols:
        dbnsfp_feature_cols = ["variant_key", *dbnsfp_feature_cols]

    dbnsfp = dbnsfp[dbnsfp_feature_cols].copy()
    dbnsfp, dbnsfp_dup = _drop_duplicate_variant_keys(dbnsfp, "dbNSFP")
    dbnsfp_keys = set(dbnsfp["variant_key"].astype(str))

    merged = merged.merge(dbnsfp, on="variant_key", how="left", validate="one_to_one")
    dbnsfp_matched = int(merged["variant_key"].isin(dbnsfp_keys).sum())
    dbnsfp_unmatched = clinvar_rows - dbnsfp_matched

    merged["has_dbnsfp_features"] = merged["variant_key"].isin(dbnsfp_keys)
    print(f"After dbNSFP merge: matched={dbnsfp_matched:,}, unmatched={dbnsfp_unmatched:,} (flagged)")
    if dbnsfp_dup:
        print(f"dbNSFP duplicate keys removed before merge: {dbnsfp_dup:,}")

    missing_dbnsfp_ratio = dbnsfp_unmatched / clinvar_rows if clinvar_rows else 0.0
    if missing_dbnsfp_ratio > 0.20:
        print(
            "WARNING: More than 20% of rows are missing dbNSFP features "
            f"({missing_dbnsfp_ratio:.2%})."
        )

    _run_sanity_checks(merged, clinvar_rows=clinvar_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)

    label_distribution = {
        "0": int((merged["label"] == 0).sum()),
        "1": int((merged["label"] == 1).sum()),
    }

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "clinvar_rows": int(clinvar_rows),
        "gnomad_matched": int(gnomad_matched),
        "gnomad_unmatched": int(gnomad_unmatched),
        "dbnsfp_matched": int(dbnsfp_matched),
        "dbnsfp_unmatched": int(dbnsfp_unmatched),
        "final_rows": int(len(merged)),
        "final_columns": int(len(merged.columns)),
        "column_list": [str(c) for c in merged.columns],
        "label_distribution": label_distribution,
        "sanity_checks_passed": True,
    }

    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Final merged dataset: {len(merged):,} rows, {len(merged.columns):,} columns")
    print(f"Saved merged parquet: {OUTPUT_PATH}")
    print(f"Saved merge metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
