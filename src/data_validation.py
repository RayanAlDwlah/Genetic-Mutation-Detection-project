#!/usr/bin/env python3
"""Post-merge dataset validation before model training.

This script validates the merged dataset and writes a comprehensive report.
Exit code:
- 0: all critical checks passed
- 1: one or more critical checks failed
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "processed" / "merged_clinvar_gnomad_dbnsfp.parquet"
FIGURES_DIR = REPO_ROOT / "results" / "figures"

CORR_FIG_PATH = FIGURES_DIR / "correlation_matrix.png"
CLASS_FIG_PATH = FIGURES_DIR / "class_distribution.png"
MISSING_FIG_PATH = FIGURES_DIR / "missing_values.png"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")


def safe_float(value: float | int | np.floating | np.integer | None) -> float | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def safe_int(value: int | np.integer | None) -> int | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return int(value)


def build_class_distribution_plot(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts(dropna=False)
    label_order = [0, 1]
    names = ["Benign (0)", "Pathogenic (1)"]
    values = [int(counts.get(k, 0)) for k in label_order]

    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.bar(names, values, color=["#4C78A8", "#F58518"])
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(CLASS_FIG_PATH, dpi=180)
    plt.close()


def build_missing_values_plot(column_quality: dict[str, dict[str, object]]) -> None:
    missing_frame = pd.DataFrame(
        {
            "column": list(column_quality.keys()),
            "missing_pct": [float(v["missing_pct"]) for v in column_quality.values()],
        }
    ).sort_values("missing_pct", ascending=False)

    plt.figure(figsize=(12, max(6, 0.28 * len(missing_frame))))
    ax = sns.barplot(data=missing_frame, x="missing_pct", y="column", color="#4C78A8")
    ax.set_title("Missing Values by Column (%)")
    ax.set_xlabel("Missing %")
    ax.set_ylabel("Column")
    ax.set_xlim(0, 100)
    plt.tight_layout()
    plt.savefig(MISSING_FIG_PATH, dpi=180)
    plt.close()


def duplicate_check(df: pd.DataFrame) -> dict[str, object]:
    dup_mask = df.duplicated(subset=["variant_key"], keep=False)
    duplicate_count = int(dup_mask.sum())
    duplicate_examples = (
        df.loc[dup_mask, ["variant_key", "chr", "pos", "ref", "alt", "gene", "label"]]
        .head(10)
        .to_dict(orient="records")
    )

    near_positions = (
        df.groupby(["chr", "pos"], dropna=False)["variant_key"]
        .nunique()
        .reset_index(name="unique_variants")
    )
    near_positions = near_positions[near_positions["unique_variants"] > 1].sort_values(
        "unique_variants", ascending=False
    )

    near_examples: list[dict[str, object]] = []
    if not near_positions.empty:
        top_pos = near_positions.head(10)
        grouped = df.groupby(["chr", "pos"], dropna=False)["variant_key"].unique()
        for _, row in top_pos.iterrows():
            key = (row["chr"], row["pos"])
            variants = grouped.get(key, [])
            near_examples.append(
                {
                    "chr": str(row["chr"]),
                    "pos": safe_int(row["pos"]),
                    "unique_variants": int(row["unique_variants"]),
                    "variant_keys_examples": [str(v) for v in list(variants)[:5]],
                }
            )

    return {
        "duplicate_variant_keys_count": duplicate_count,
        "duplicate_variant_keys_examples": duplicate_examples,
        "near_duplicate_positions_count": int(len(near_positions)),
        "near_duplicate_positions_examples": near_examples,
    }


def label_integrity_check(df: pd.DataFrame) -> dict[str, object]:
    labels = df["label"]

    nan_count = int(labels.isna().sum())
    invalid_mask = labels.notna() & ~labels.isin([0, 1])
    invalid_count = int(invalid_mask.sum())
    invalid_values = sorted({str(v) for v in labels[invalid_mask].unique()})

    label_by_variant = df.groupby("variant_key", dropna=False)["label"].nunique(dropna=True)
    conflict_keys = label_by_variant[label_by_variant > 1].index
    conflict_count = int(len(conflict_keys))

    if conflict_count > 0:
        conflict_examples = (
            df[df["variant_key"].isin(conflict_keys)][["variant_key", "label", "gene", "chr", "pos"]]
            .sort_values(["variant_key", "label"])
            .head(20)
            .to_dict(orient="records")
        )
    else:
        conflict_examples = []

    return {
        "allowed_labels": [0, 1],
        "nan_label_count": nan_count,
        "invalid_label_count": invalid_count,
        "invalid_label_values": invalid_values,
        "variant_keys_with_both_labels_count": conflict_count,
        "variant_keys_with_both_labels_examples": conflict_examples,
    }


def column_quality_check(df: pd.DataFrame) -> dict[str, object]:
    total_rows = len(df)
    quality: dict[str, dict[str, object]] = {}

    all_nan_cols: list[str] = []
    high_missing_cols: list[str] = []
    zero_variance_cols: list[str] = []

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_pct = (missing_count / total_rows * 100.0) if total_rows else 0.0
        non_null_nunique = int(series.nunique(dropna=True))
        is_zero_variance = non_null_nunique <= 1

        quality[col] = {
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 4),
            "non_null_unique_values": non_null_nunique,
            "zero_variance": bool(is_zero_variance),
        }

        if missing_count == total_rows:
            all_nan_cols.append(col)
        if missing_pct > 80.0:
            high_missing_cols.append(col)
        if is_zero_variance:
            zero_variance_cols.append(col)

    return {
        "per_column": quality,
        "all_nan_columns": all_nan_cols,
        "high_missing_columns_gt80pct": high_missing_cols,
        "zero_variance_columns": zero_variance_cols,
    }


def outlier_check(df: pd.DataFrame) -> dict[str, object]:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

    per_column: dict[str, dict[str, object]] = {}
    outlier_flagged: list[str] = []

    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()

        if valid.empty:
            per_column[col] = {
                "count_non_null": 0,
                "mean": None,
                "std": None,
                "min": None,
                "q1": None,
                "q3": None,
                "max": None,
                "iqr": None,
                "lower_5iqr": None,
                "upper_5iqr": None,
                "outlier_count": 0,
                "outlier_pct": 0.0,
            }
            continue

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 5 * iqr
        upper = q3 + 5 * iqr
        outlier_mask = (valid < lower) | (valid > upper)
        outlier_count = int(outlier_mask.sum())

        if outlier_count > 0:
            outlier_flagged.append(col)

        per_column[col] = {
            "count_non_null": int(valid.shape[0]),
            "mean": safe_float(valid.mean()),
            "std": safe_float(valid.std()),
            "min": safe_float(valid.min()),
            "q1": safe_float(q1),
            "q3": safe_float(q3),
            "max": safe_float(valid.max()),
            "iqr": safe_float(iqr),
            "lower_5iqr": safe_float(lower),
            "upper_5iqr": safe_float(upper),
            "outlier_count": outlier_count,
            "outlier_pct": round(outlier_count / valid.shape[0] * 100.0, 4),
        }

    return {
        "numeric_columns_checked": numeric_cols,
        "columns_with_outliers": outlier_flagged,
        "per_column": per_column,
    }


def class_balance_check(df: pd.DataFrame) -> dict[str, object]:
    counts = df["label"].value_counts(dropna=False)
    benign = int(counts.get(0, 0))
    pathogenic = int(counts.get(1, 0))

    ratio = None
    warning = False
    recommendation = "Class ratio is within 1:3 and 3:1 bounds."

    if benign == 0 or pathogenic == 0:
        warning = True
        recommendation = "Severe class imbalance detected. Use Focal Loss or class_weight='balanced'."
    else:
        ratio = pathogenic / benign
        if ratio < (1 / 3) or ratio > 3:
            warning = True
            recommendation = "Class imbalance warning: consider Focal Loss or class_weight='balanced'."

    return {
        "benign_count": benign,
        "pathogenic_count": pathogenic,
        "pathogenic_to_benign_ratio": None if ratio is None else round(ratio, 6),
        "warning": warning,
        "recommendation": recommendation,
    }


def gene_distribution_check(df: pd.DataFrame) -> dict[str, object]:
    gene_counts = df["gene"].value_counts(dropna=True)
    unique_genes = int(gene_counts.shape[0])

    single_variant_genes = gene_counts[gene_counts == 1]

    by_gene = df.groupby("gene", dropna=True)["label"].agg(["nunique", "min", "max", "count"])
    only_pathogenic = by_gene[(by_gene["nunique"] == 1) & (by_gene["min"] == 1)]
    only_benign = by_gene[(by_gene["nunique"] == 1) & (by_gene["max"] == 0)]

    return {
        "unique_genes": unique_genes,
        "top_10_genes": {str(k): int(v) for k, v in gene_counts.head(10).to_dict().items()},
        "genes_with_single_variant_count": int(single_variant_genes.shape[0]),
        "genes_with_single_variant_examples": [str(g) for g in single_variant_genes.head(20).index.tolist()],
        "genes_only_pathogenic_count": int(only_pathogenic.shape[0]),
        "genes_only_pathogenic_examples": [str(g) for g in only_pathogenic.head(20).index.tolist()],
        "genes_only_benign_count": int(only_benign.shape[0]),
        "genes_only_benign_examples": [str(g) for g in only_benign.head(20).index.tolist()],
    }


def correlation_check(
    df: pd.DataFrame,
    missing_pct_map: dict[str, float],
) -> dict[str, object]:
    numeric_df = df.select_dtypes(include=[np.number, bool]).copy()

    if numeric_df.empty:
        plt.figure(figsize=(6, 4))
        plt.title("Correlation Matrix (No Numeric Columns)")
        plt.tight_layout()
        plt.savefig(CORR_FIG_PATH, dpi=180)
        plt.close()
        return {
            "numeric_columns": [],
            "high_correlation_pairs_abs_gt_0_95": [],
        }

    for col in numeric_df.select_dtypes(include=[bool]).columns:
        numeric_df[col] = numeric_df[col].astype(int)

    corr = numeric_df.corr(numeric_only=True)

    size = max(8, min(30, int(0.5 * max(1, len(corr.columns)))))
    plt.figure(figsize=(size, size))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, center=0, square=True)
    plt.title("Numeric Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(CORR_FIG_PATH, dpi=180)
    plt.close()

    high_pairs: list[dict[str, object]] = []
    columns = list(corr.columns)
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            c1 = columns[i]
            c2 = columns[j]
            value = corr.iloc[i, j]
            if pd.isna(value):
                continue
            if abs(value) > 0.95:
                miss1 = float(missing_pct_map.get(c1, 0.0))
                miss2 = float(missing_pct_map.get(c2, 0.0))
                if miss1 > miss2:
                    suggest_drop = c1
                elif miss2 > miss1:
                    suggest_drop = c2
                else:
                    suggest_drop = sorted([c1, c2])[1]

                high_pairs.append(
                    {
                        "feature_1": c1,
                        "feature_2": c2,
                        "correlation": round(float(value), 6),
                        "abs_correlation": round(float(abs(value)), 6),
                        "feature_1_missing_pct": round(miss1, 4),
                        "feature_2_missing_pct": round(miss2, 4),
                        "suggest_drop": suggest_drop,
                    }
                )

    high_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)

    return {
        "numeric_columns": columns,
        "high_correlation_pairs_abs_gt_0_95": high_pairs,
    }


def print_report(report: dict[str, object], critical_issues: list[str]) -> None:
    print("\n" + "=" * 88)
    print("VALIDATION REPORT")
    print("=" * 88)

    overview = report["overview"]
    print(f"Rows: {overview['rows']:,} | Columns: {overview['columns']:,}")

    dup = report["duplicate_check"]
    print("\n[1] Duplicate Check")
    print(f"- duplicate variant_keys: {dup['duplicate_variant_keys_count']:,}")
    print(f"- near-duplicate positions (same chr:pos, >1 variant): {dup['near_duplicate_positions_count']:,}")

    labels = report["label_integrity"]
    print("\n[2] Label Integrity")
    print(f"- NaN labels: {labels['nan_label_count']:,}")
    print(f"- invalid labels: {labels['invalid_label_count']:,}")
    print(f"- variant_keys with both labels: {labels['variant_keys_with_both_labels_count']:,}")

    column_q = report["column_quality"]
    print("\n[3] Column Quality")
    print(f"- all-NaN columns: {len(column_q['all_nan_columns'])}")
    print(f"- >80% missing columns: {len(column_q['high_missing_columns_gt80pct'])}")
    print(f"- zero variance columns: {len(column_q['zero_variance_columns'])}")

    outliers = report["outlier_detection"]
    print("\n[4] Outlier Detection")
    print(f"- numeric columns checked: {len(outliers['numeric_columns_checked'])}")
    print(f"- columns with potential outliers (>5*IQR): {len(outliers['columns_with_outliers'])}")

    cls = report["class_balance"]
    print("\n[5] Class Balance")
    ratio = cls["pathogenic_to_benign_ratio"]
    ratio_str = "NA" if ratio is None else f"{ratio:.4f}"
    print(f"- benign(0): {cls['benign_count']:,} | pathogenic(1): {cls['pathogenic_count']:,}")
    print(f"- pathogenic:benign ratio: {ratio_str}")
    if cls["warning"]:
        print(f"- WARNING: {cls['recommendation']}")

    gene = report["gene_distribution"]
    print("\n[6] Gene Distribution")
    print(f"- unique genes: {gene['unique_genes']:,}")
    print(f"- genes with single variant: {gene['genes_with_single_variant_count']:,}")
    print(f"- genes only pathogenic: {gene['genes_only_pathogenic_count']:,}")
    print(f"- genes only benign: {gene['genes_only_benign_count']:,}")

    corr = report["feature_correlation"]
    print("\n[7] Feature Correlation")
    print(
        "- pairs with |corr| > 0.95: "
        f"{len(corr['high_correlation_pairs_abs_gt_0_95']):,}"
    )

    print("\nCritical Issues")
    if critical_issues:
        for issue in critical_issues:
            print(f"- {issue}")
    else:
        print("- none")

    print("\nArtifacts")
    print(f"- Correlation heatmap: {CORR_FIG_PATH}")
    print(f"- Class distribution plot: {CLASS_FIG_PATH}")
    print(f"- Missing values plot: {MISSING_FIG_PATH}")


def main() -> None:
    require_file(INPUT_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH)

    report: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(INPUT_PATH),
        "overview": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        },
    }

    report["duplicate_check"] = duplicate_check(df)
    report["label_integrity"] = label_integrity_check(df)
    report["column_quality"] = column_quality_check(df)
    report["outlier_detection"] = outlier_check(df)
    report["class_balance"] = class_balance_check(df)
    report["gene_distribution"] = gene_distribution_check(df)

    missing_pct_map = {
        col: float(details["missing_pct"])
        for col, details in report["column_quality"]["per_column"].items()
    }
    report["feature_correlation"] = correlation_check(df, missing_pct_map=missing_pct_map)

    build_class_distribution_plot(df)
    build_missing_values_plot(report["column_quality"]["per_column"])

    critical_issues: list[str] = []
    labels = report["label_integrity"]
    colq = report["column_quality"]

    if labels["nan_label_count"] > 0 or labels["invalid_label_count"] > 0:
        critical_issues.append("Label integrity violation: labels must be only 0/1 with no NaN.")

    if labels["variant_keys_with_both_labels_count"] > 0:
        critical_issues.append("Duplicate labels detected: some variant_keys have both label=0 and label=1.")

    if len(colq["all_nan_columns"]) > 0:
        critical_issues.append("One or more columns are 100% NaN.")

    report["critical_issues"] = {
        "count": len(critical_issues),
        "items": critical_issues,
        "critical_passed": len(critical_issues) == 0,
    }

    print_report(report, critical_issues)

    sys.exit(0 if len(critical_issues) == 0 else 1)


if __name__ == "__main__":
    main()
