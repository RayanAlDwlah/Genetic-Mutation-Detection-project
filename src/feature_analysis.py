#!/usr/bin/env python3
"""Feature analysis, filtering, and final dataset versioning.

This version performs all analysis in-memory and writes only parquet outputs.
No JSON metadata files are produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from src.utils import require_file


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "processed" / "merged_clinvar_gnomad_dbnsfp.parquet"

STRICT_OUTPUT = REPO_ROOT / "data" / "processed" / "final_strict.parquet"
BALANCED_OUTPUT = REPO_ROOT / "data" / "processed" / "final_balanced.parquet"

CONSERVATION_PRIORITY = {
    "phylop": 0,
    "phastcons": 1,
    "gerp": 2,
}
SUBSTITUTION_PRIORITY = {
    "grantham": 0,
    "blosum": 1,
}

CONSERVATION_FEATURES = [
    "phyloP100way_vertebrate",
    "phyloP30way_mammalian",
    "phastCons100way_vertebrate",
    "phastCons30way_mammalian",
    "GERP++_RS",
    "GERP++_NR",
    "SiPhy_29way_logOdds",
]

NON_FEATURE_COLUMNS = {
    "variant_key",
    "chr",
    "pos",
    "ref",
    "alt",
    "gene",
    "label",
    "review_stars",
    "ClinicalSignificance",
    "PhenotypeIDS",
}

IMPORTANT_IMPUTE_FLAGS = [
    "AF",
    "AF_popmax",
    "log_AF",
    "phyloP100way_vertebrate",
    "phastCons100way_vertebrate",
    "GERP++_RS",
    "Grantham_distance",
    "BLOSUM62_score",
]

def load_input() -> pd.DataFrame:
    require_file(INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)

    if "label" not in df.columns:
        raise ValueError("Input dataset missing required 'label' column")
    if "review_stars" not in df.columns:
        raise ValueError("Input dataset missing required 'review_stars' column")

    return df


def is_conservation_feature(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in CONSERVATION_PRIORITY)


def is_substitution_feature(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in SUBSTITUTION_PRIORITY)


def conservation_rank(name: str) -> int | None:
    n = name.lower()
    for key, rank in CONSERVATION_PRIORITY.items():
        if key in n:
            return rank
    return None


def substitution_rank(name: str) -> int | None:
    n = name.lower()
    for key, rank in SUBSTITUTION_PRIORITY.items():
        if key in n:
            return rank
    return None


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def step1_drop_flagged_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    all_nan = [c for c in df.columns if df[c].isna().all()]
    zero_var = [c for c in df.columns if df[c].nunique(dropna=True) <= 1 and c not in all_nan]

    to_drop = all_nan + zero_var
    if to_drop:
        df = df.drop(columns=to_drop)

    print("STEP 1 — Columns dropped from quality checks")
    print(f"- 100% NaN columns dropped: {all_nan if all_nan else 'none'}")
    print(f"- zero-variance columns dropped: {zero_var if zero_var else 'none'}")

    return df, all_nan, zero_var


def compute_high_corr_pairs(df: pd.DataFrame, threshold: float = 0.95) -> list[dict[str, Any]]:
    feature_cols = get_feature_columns(df)

    numeric_cols: list[str] = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            numeric_cols.append(col)

    if not numeric_cols:
        return []

    numeric_df = df[numeric_cols].copy()
    for col in numeric_df.columns:
        if pd.api.types.is_bool_dtype(numeric_df[col]):
            numeric_df[col] = numeric_df[col].astype(int)

    corr = numeric_df.corr()

    pairs: list[dict[str, Any]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.isna(v):
                continue
            if abs(v) > threshold:
                pairs.append(
                    {
                        "feature_1": cols[i],
                        "feature_2": cols[j],
                        "correlation": float(v),
                    }
                )

    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return pairs


def choose_drop_feature(
    col_a: str,
    col_b: str,
    missing_pct: dict[str, float],
) -> tuple[str, str, str]:
    miss_a = missing_pct.get(col_a, 0.0)
    miss_b = missing_pct.get(col_b, 0.0)

    if miss_a > miss_b:
        return col_a, col_b, "higher missing"
    if miss_b > miss_a:
        return col_b, col_a, "higher missing"

    if is_conservation_feature(col_a) and is_conservation_feature(col_b):
        rank_a = conservation_rank(col_a)
        rank_b = conservation_rank(col_b)
        if rank_a is not None and rank_b is not None and rank_a != rank_b:
            if rank_a < rank_b:
                return col_b, col_a, "conservation priority"
            return col_a, col_b, "conservation priority"

    if is_substitution_feature(col_a) and is_substitution_feature(col_b):
        rank_a = substitution_rank(col_a)
        rank_b = substitution_rank(col_b)
        if rank_a is not None and rank_b is not None and rank_a != rank_b:
            if rank_a < rank_b:
                return col_b, col_a, "substitution priority"
            return col_a, col_b, "substitution priority"

    # Deterministic fallback for ties.
    drop = sorted([col_a, col_b])[1]
    keep = col_b if drop == col_a else col_a
    return drop, keep, "deterministic fallback"


def step2_drop_correlated_columns(df: pd.DataFrame, corr_pairs: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    dropped_records: list[dict[str, Any]] = []

    print("\nSTEP 2 — Correlation filtering")

    for pair in corr_pairs:
        col_a = pair.get("feature_1")
        col_b = pair.get("feature_2")
        corr = float(pair.get("correlation", 0.0))

        if not isinstance(col_a, str) or not isinstance(col_b, str):
            continue
        if col_a not in df.columns or col_b not in df.columns:
            continue

        missing_pct = {
            col_a: float(df[col_a].isna().mean() * 100.0),
            col_b: float(df[col_b].isna().mean() * 100.0),
        }

        drop_col, keep_col, reason = choose_drop_feature(col_a, col_b, missing_pct)
        if drop_col not in df.columns:
            continue

        df = df.drop(columns=[drop_col])
        rec = {
            "dropped": drop_col,
            "kept": keep_col,
            "correlation": round(corr, 6),
            "reason": reason,
        }
        dropped_records.append(rec)

        print(f"- Dropped {drop_col} due to r={corr:.2f} correlation with {keep_col} ({reason})")

    if not dropped_records:
        print("- No correlated columns dropped")

    return df, dropped_records


def make_strict_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    strict = df[df["review_stars"] >= 2].copy()
    rows_start = len(strict)

    conservation_cols = [c for c in CONSERVATION_FEATURES if c in strict.columns]
    if conservation_cols:
        strict = strict.dropna(subset=conservation_cols)

    feature_cols = get_feature_columns(strict)
    other_feature_cols = [c for c in feature_cols if c not in set(conservation_cols)]

    if other_feature_cols:
        row_missing_other = strict[other_feature_cols].isna().sum(axis=1)
        strict = strict[row_missing_other <= 3].copy()

    missing_after = float(strict.isna().mean().max() * 100.0) if len(strict.columns) else 0.0

    info = {
        "rows_start": int(rows_start),
        "rows_after": int(len(strict)),
        "conservation_features": conservation_cols,
        "max_missing_pct_after_filters": round(missing_after, 4),
        "missing_policy": "zero tolerance for conservation features",
    }

    return strict, info


def make_balanced_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    balanced = df[df["review_stars"] >= 1].copy()
    feature_cols = get_feature_columns(balanced)

    missing_pct = (balanced[feature_cols].isna().mean() * 100.0) if feature_cols else pd.Series(dtype=float)
    drop_over_20 = [c for c, v in missing_pct.items() if v > 20.0]

    if drop_over_20:
        balanced = balanced.drop(columns=drop_over_20)

    feature_cols_after_drop = get_feature_columns(balanced)
    pre_impute_missing_pct = (balanced[feature_cols_after_drop].isna().mean() * 100.0) if feature_cols_after_drop else pd.Series(dtype=float)

    imputed_columns: list[str] = []

    for col in feature_cols_after_drop:
        s = balanced[col]
        missing_mask = s.isna()
        if not missing_mask.any():
            continue

        if pd.api.types.is_numeric_dtype(s):
            fill_value = s.median(skipna=True)
            if pd.isna(fill_value):
                fill_value = 0.0
            balanced[col] = s.fillna(fill_value)
        else:
            mode = s.mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "UNKNOWN"
            balanced[col] = s.fillna(fill_value)

        imputed_columns.append(col)

        if col in IMPORTANT_IMPUTE_FLAGS:
            balanced[f"is_imputed_{col}"] = missing_mask.astype(bool)

    info = {
        "rows": int(len(balanced)),
        "columns_dropped_over_20pct_missing": drop_over_20,
        "max_missing_pct_pre_imputation": round(float(pre_impute_missing_pct.max()) if len(pre_impute_missing_pct) else 0.0, 4),
        "imputed_columns": imputed_columns,
        "imputation_method": "median for numeric, mode for categorical",
        "missing_policy": "up to 20% allowed, then imputed",
    }

    return balanced, info


def label_summary(df: pd.DataFrame) -> tuple[int, int, float | None]:
    benign = int((df["label"] == 0).sum())
    pathogenic = int((df["label"] == 1).sum())
    ratio = None if benign == 0 else pathogenic / benign
    return benign, pathogenic, ratio


def version_summary(name: str, df: pd.DataFrame, pre_impute_max_missing: float | None = None) -> dict[str, Any]:
    feature_cols = get_feature_columns(df)
    benign, pathogenic, ratio = label_summary(df)

    missing_pct_per_col = (df.isna().mean() * 100.0).sort_values(ascending=False)
    max_missing_pct = float(missing_pct_per_col.max()) if len(missing_pct_per_col) else 0.0

    print(f"\n{name.upper()} VERSION SUMMARY")
    print(f"- rows: {len(df):,}")
    print(f"- total columns: {len(df.columns):,}")
    print(f"- feature columns: {len(feature_cols):,}")
    print(f"- label distribution: benign={benign:,}, pathogenic={pathogenic:,}, ratio={ratio if ratio is not None else 'NA'}")
    if pre_impute_max_missing is not None:
        print(f"- max missing % before imputation: {pre_impute_max_missing:.4f}")
    print(f"- max missing % after processing: {max_missing_pct:.4f}")
    print("- feature dtypes:")
    for col in feature_cols:
        print(f"  {col}: {df[col].dtype}")

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "feature_columns": int(len(feature_cols)),
        "pathogenic": pathogenic,
        "benign": benign,
        "label_ratio_pathogenic_to_benign": None if ratio is None else round(float(ratio), 6),
        "max_missing_pct_after_processing": round(max_missing_pct, 6),
        "feature_dtypes": {col: str(df[col].dtype) for col in feature_cols},
    }


def main() -> None:
    df = load_input()

    print("Loaded input dataset")
    print(f"- path: {INPUT_PATH}")
    print(f"- rows: {len(df):,}, columns: {len(df.columns):,}")

    # Step 1
    df, _, _ = step1_drop_flagged_columns(df)

    # Step 2
    corr_pairs = compute_high_corr_pairs(df, threshold=0.95)
    df, _ = step2_drop_correlated_columns(df, corr_pairs)

    # Step 3A
    strict_df, strict_info = make_strict_dataset(df)
    STRICT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    strict_df.to_parquet(STRICT_OUTPUT, index=False)

    # Step 3B
    balanced_df, balanced_info = make_balanced_dataset(df)
    BALANCED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_parquet(BALANCED_OUTPUT, index=False)

    # Step 4 summaries
    version_summary("strict", strict_df)
    version_summary(
        "balanced",
        balanced_df,
        pre_impute_max_missing=balanced_info["max_missing_pct_pre_imputation"],
    )

    print("\nSaved outputs")
    print(f"- strict dataset: {STRICT_OUTPUT}")
    print(f"- balanced dataset: {BALANCED_OUTPUT}")
    print("- metadata: skipped (documented in notebooks)")


if __name__ == "__main__":
    main()
