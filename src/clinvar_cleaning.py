#!/usr/bin/env python3
"""Clean ClinVar data and build binary labels for mutation classification."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any
from output import echo

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit("Missing dependency: pandas. Install requirements first.") from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit("Missing dependency: pyyaml. Install requirements first.") from exc


def resolve_path(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def normalize_chromosome(value: Any) -> str | None:
    text = normalize_text(value)
    if not text:
        return None

    if text.startswith("chr"):
        text = text[3:]

    text = text.upper()
    if text == "23":
        return "X"
    if text == "24":
        return "Y"

    if text.isdigit():
        return str(int(text))

    if text in {"X", "Y"}:
        return text

    return text


def normalize_allele_series(series: pd.Series) -> pd.Series:
    out = series.fillna("").astype(str).str.strip().str.upper()
    return out.replace({"": "", "NA": "", "NAN": "", "NONE": "", "NULL": "", ".": "", "-": ""})


def extract_review_stars(value: Any) -> int:
    text = normalize_text(value)
    if not text:
        return 0

    if "practice guideline" in text:
        return 4
    if "reviewed by expert panel" in text:
        return 3
    if "multiple submitters" in text and "no conflicts" in text and "criteria provided" in text:
        return 2
    if "criteria provided" in text and (
        "single submitter" in text
        or "conflicting interpretations" in text
        or "multiple submitters" in text
    ):
        return 1

    return 0


def chromosome_sort_key(chrom: str) -> tuple[int, int | str]:
    if chrom.isdigit():
        return (0, int(chrom))
    if chrom == "X":
        return (1, 23)
    if chrom == "Y":
        return (1, 24)
    return (2, chrom)


def print_before_after(step_name: str, before_rows: int, after_rows: int) -> None:
    echo(f"{step_name}: {before_rows:,} -> {after_rows:,}")


def print_series_counts(title: str, series: pd.Series) -> None:
    echo(title)
    if series.empty:
        echo("  (empty)")
        return
    for key, value in series.items():
        echo(f"  {key}: {int(value):,}")


def print_bar(label: str, count: int, total: int, width: int = 40) -> None:
    pct = (count / total * 100.0) if total else 0.0
    filled = int(round((pct / 100.0) * width))
    bar = "#" * filled + "-" * (width - filled)
    echo(f"  {label:<10} {count:>10,} ({pct:6.2f}%) |{bar}|")


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a top-level mapping")
    return data


def pick_input_file(repo_root: Path, config: dict[str, Any]) -> Path:
    cleaning_cfg = config.get("clinvar_cleaning", {}) or {}
    source_cfg = config.get("data_sources", {}).get("clinvar", {}) or {}

    candidates = [
        cleaning_cfg.get("input_file"),
        "data/raw/variant_summary.txt.gz",
        source_cfg.get("file"),
        "data/raw/clinvar/variant_summary.txt.gz",
    ]

    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)

    for candidate in ordered:
        resolved = resolve_path(repo_root, candidate)
        if resolved.exists():
            return resolved

    if not ordered:
        raise FileNotFoundError("No ClinVar input file is configured")

    # Return first configured candidate to produce a precise error path.
    return resolve_path(repo_root, ordered[0])


def run_pipeline(config_path: Path, strict: bool) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(config_path)
    cleaning_cfg = config.get("clinvar_cleaning", {}) or {}

    if not cleaning_cfg:
        raise ValueError("Missing 'clinvar_cleaning' section in configs/config.yaml")

    input_path = pick_input_file(repo_root, config)
    output_path = resolve_path(
        repo_root,
        cleaning_cfg.get("output_file", "data/intermediate/clinvar_labeled_clean.parquet"),
    )

    if not input_path.exists():
        raise FileNotFoundError(f"ClinVar input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    variant_type_target = normalize_text(cleaning_cfg.get("variant_type", "single nucleotide variant"))
    mc_column = cleaning_cfg.get("molecular_consequence_column", "MolecularConsequence")
    mc_target = normalize_text(cleaning_cfg.get("molecular_consequence", "missense_variant"))
    mc_non_empty_min = float(cleaning_cfg.get("molecular_consequence_non_empty_ratio_min", 0.1))

    labels_kept = cleaning_cfg.get(
        "labels_kept",
        ["Pathogenic", "Likely pathogenic", "Benign", "Likely benign"],
    )
    positive_labels = cleaning_cfg.get("label_positive", ["Pathogenic", "Likely pathogenic"])
    negative_labels = cleaning_cfg.get("label_negative", ["Benign", "Likely benign"])
    configured_min_review_stars = int(cleaning_cfg.get("min_review_stars", 1))
    min_review_stars = 2 if strict else configured_min_review_stars

    allowed_chromosomes = set(
        str(item).upper().replace("CHR", "")
        for item in cleaning_cfg.get(
            "allowed_chromosomes",
            [str(idx) for idx in range(1, 23)] + ["X", "Y"],
        )
    )

    echo("=== STEP 1: Load raw ClinVar ===")
    echo(f"Source file: {input_path}")
    df = pd.read_csv(input_path, sep="\t", low_memory=False)
    echo(f"Total rows: {len(df):,}")
    echo(f"Total columns: {df.shape[1]:,}")
    echo("Column names:")
    for col in df.columns.tolist():
        echo(f"  - {col}")
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    echo(f"Memory usage: {memory_mb:,.2f} MB")

    echo("\n=== STEP 2: Filter variant type ===")
    if "Type" not in df.columns:
        raise KeyError("Required column is missing: Type")
    before = len(df)
    df = df[df["Type"].map(normalize_text) == variant_type_target].copy()
    print_before_after("Rows", before, len(df))

    echo("\n=== STEP 3: Filter molecular consequence ===")
    before = len(df)
    molecular_filter_applied = True
    molecular_fallback_reason = ""

    if mc_column not in df.columns:
        molecular_filter_applied = False
        molecular_fallback_reason = (
            f"Column '{mc_column}' is missing. Falling back to Type-only filtering."
        )
    else:
        mc_series = df[mc_column]
        non_empty_ratio = (
            mc_series.notna() & mc_series.astype(str).str.strip().ne("")
        ).mean()
        if non_empty_ratio < mc_non_empty_min:
            molecular_filter_applied = False
            molecular_fallback_reason = (
                f"Column '{mc_column}' is mostly empty "
                f"(non-empty ratio={non_empty_ratio:.2%}). Falling back to Type-only filtering."
            )

    if molecular_filter_applied:
        pattern = rf"(?:^|,\s*){re.escape(mc_target)}(?:\s*,|$)"
        mask = (
            df[mc_column]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(pattern, regex=True)
        )
        df = df[mask].copy()
        print_before_after("Rows", before, len(df))
    else:
        print_before_after("Rows", before, len(df))
        echo(f"Fallback note: {molecular_fallback_reason}")

    echo("\n=== STEP 4: Filter clinical significance ===")
    if "ClinicalSignificance" not in df.columns:
        raise KeyError("Required column is missing: ClinicalSignificance")

    canonical_by_normalized = {normalize_text(value): str(value) for value in labels_kept}
    allowed_values = set(canonical_by_normalized.keys())

    before = len(df)
    clinical_norm = df["ClinicalSignificance"].map(normalize_text)
    df = df[clinical_norm.isin(allowed_values)].copy()
    df["ClinicalSignificance"] = clinical_norm[clinical_norm.isin(allowed_values)].map(canonical_by_normalized)
    print_before_after("Rows", before, len(df))
    print_series_counts(
        "ClinicalSignificance counts:",
        df["ClinicalSignificance"].value_counts().sort_values(ascending=False),
    )

    echo("\n=== STEP 5: Create binary label ===")
    positive_set = {normalize_text(v) for v in positive_labels}
    negative_set = {normalize_text(v) for v in negative_labels}

    label_map: dict[str, int] = {}
    for value in labels_kept:
        normalized = normalize_text(value)
        if normalized in positive_set:
            label_map[normalized] = 1
        elif normalized in negative_set:
            label_map[normalized] = 0

    df["label"] = df["ClinicalSignificance"].map(normalize_text).map(label_map)
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype("int8")

    class_counts = df["label"].value_counts().sort_index()
    total_labeled = int(class_counts.sum())
    echo("Class distribution:")
    for label in [0, 1]:
        count = int(class_counts.get(label, 0))
        pct = (count / total_labeled * 100.0) if total_labeled else 0.0
        echo(f"  label={label}: {count:,} ({pct:.2f}%)")

    echo("\n=== STEP 6: Filter by review quality ===")
    if "ReviewStatus" not in df.columns:
        echo("ReviewStatus column is missing. Setting review_stars=0 for all rows.")
        df["review_stars"] = 0
    else:
        df["review_stars"] = df["ReviewStatus"].map(extract_review_stars).astype("int8")

    print_series_counts(
        "Review star distribution (before threshold):",
        df["review_stars"].value_counts().sort_index(),
    )

    before = len(df)
    df = df[df["review_stars"] >= min_review_stars].copy()
    print_before_after("Rows", before, len(df))

    echo("\n=== STEP 7: Standardize columns ===")
    required_cols = ["Chromosome", "GeneSymbol"]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise KeyError(f"Required columns are missing: {missing_required}")

    if "MolecularConsequence" not in df.columns:
        df["MolecularConsequence"] = pd.NA
    if "PhenotypeIDS" not in df.columns:
        df["PhenotypeIDS"] = pd.NA

    before = len(df)
    df["chr"] = df["Chromosome"].map(normalize_chromosome)
    df = df[df["chr"].isin(allowed_chromosomes)].copy()
    print_before_after("Rows after chromosome normalization/filter", before, len(df))

    pos_vcf = pd.to_numeric(df["PositionVCF"], errors="coerce") if "PositionVCF" in df.columns else pd.Series(pd.NA, index=df.index, dtype="float64")
    pos_start = pd.to_numeric(df["Start"], errors="coerce") if "Start" in df.columns else pd.Series(pd.NA, index=df.index, dtype="float64")
    df["pos"] = pos_vcf.where(pos_vcf.notna(), pos_start)

    ref_vcf = normalize_allele_series(df["ReferenceAlleleVCF"]) if "ReferenceAlleleVCF" in df.columns else pd.Series("", index=df.index)
    ref_raw = normalize_allele_series(df["ReferenceAllele"]) if "ReferenceAllele" in df.columns else pd.Series("", index=df.index)
    alt_vcf = normalize_allele_series(df["AlternateAlleleVCF"]) if "AlternateAlleleVCF" in df.columns else pd.Series("", index=df.index)
    alt_raw = normalize_allele_series(df["AlternateAllele"]) if "AlternateAllele" in df.columns else pd.Series("", index=df.index)

    df["ref"] = ref_vcf.where(ref_vcf.ne(""), ref_raw)
    df["alt"] = alt_vcf.where(alt_vcf.ne(""), alt_raw)
    df["gene"] = df["GeneSymbol"].fillna("").astype(str).str.strip()

    before = len(df)
    df = df[
        df["pos"].notna()
        & df["ref"].ne("")
        & df["alt"].ne("")
        & df["gene"].ne("")
        & df["ref"].str.len().eq(1)
        & df["alt"].str.len().eq(1)
    ].copy()
    df["pos"] = df["pos"].astype("int64")
    print_before_after("Rows after required-field cleanup", before, len(df))

    df["variant_key"] = (
        df["chr"]
        + ":"
        + df["pos"].astype(str)
        + ":"
        + df["ref"]
        + ":"
        + df["alt"]
    )

    keep_columns = [
        "chr",
        "pos",
        "ref",
        "alt",
        "gene",
        "label",
        "review_stars",
        "variant_key",
        "ClinicalSignificance",
        "MolecularConsequence",
        "PhenotypeIDS",
    ]
    df = df[keep_columns].copy()

    echo("\n=== STEP 8: Remove duplicates and conflicts ===")
    conflict_key_mask = df.groupby("variant_key")["label"].nunique() > 1
    conflict_keys = set(conflict_key_mask[conflict_key_mask].index.tolist())
    conflicting_labels_removed = int(df["variant_key"].isin(conflict_keys).sum())

    if conflict_keys:
        df = df[~df["variant_key"].isin(conflict_keys)].copy()

    duplicate_rows_found = int(df.duplicated(subset=["variant_key"], keep="first").sum())
    before = len(df)
    df = df.drop_duplicates(subset=["variant_key"], keep="first").copy()
    duplicates_removed = before - len(df)

    echo(f"Duplicate rows found: {duplicate_rows_found:,}")
    echo(f"Conflicting label rows removed: {conflicting_labels_removed:,}")
    echo(f"Final row count after deduplication: {len(df):,}")

    echo("\n=== STEP 9: Save output ===")
    df.to_parquet(output_path, index=False)

    pathogenic_count = int((df["label"] == 1).sum())
    benign_count = int((df["label"] == 0).sum())
    total_rows = int(len(df))

    echo(f"Saved parquet: {output_path}")

    echo("\n=== STEP 10: Final summary ===")
    echo(f"Total clean variants: {total_rows:,}")

    echo("Class distribution (Pathogenic vs Benign):")
    print_bar("Pathogenic", pathogenic_count, total_rows)
    print_bar("Benign", benign_count, total_rows)

    echo("Top 20 genes by variant count:")
    gene_counts = df["gene"].value_counts().head(20)
    if gene_counts.empty:
        echo("  (empty)")
    else:
        echo(gene_counts.to_string())

    echo("Chromosome distribution:")
    chromosome_counts = df["chr"].value_counts()
    if chromosome_counts.empty:
        echo("  (empty)")
    else:
        for chrom in sorted(chromosome_counts.index.tolist(), key=chromosome_sort_key):
            echo(f"  {chrom}: {int(chromosome_counts[chrom]):,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean ClinVar and create binary labels")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use min_review_stars=2 instead of configured value",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(repo_root, args.config)

    try:
        run_pipeline(config_path=config_path, strict=args.strict)
    except Exception as exc:  # pragma: no cover - CLI error path
        echo(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
