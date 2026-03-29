#!/usr/bin/env python3
"""Gene-level train/val/test splitting with strict overlap validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.utils import resolve_path


DEFAULT_INPUT = "data/processed/final_balanced.parquet"
DEFAULT_OUTPUT_DIR = "data/splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gene-level data splitting with GroupShuffleSplit")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input parquet path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory path")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def get_gene_column(df: pd.DataFrame) -> str:
    if "GeneSymbol" in df.columns:
        return "GeneSymbol"
    if "gene" in df.columns:
        return "gene"
    raise ValueError("Input dataset must contain either 'gene' or 'GeneSymbol' column")


def label_stats(df: pd.DataFrame) -> tuple[int, int, float]:
    pathogenic = int((df["label"] == 1).sum())
    benign = int((df["label"] == 0).sum())
    total = len(df)
    ratio = (pathogenic / total) if total else 0.0
    return pathogenic, benign, ratio


def gene_preview(genes: list[str], limit: int = 10) -> list[str]:
    unique = sorted(set(genes))
    if len(unique) <= limit:
        return unique
    return [*unique[:limit], f"and {len(unique) - limit} more"]


def split_dataframe(
    df: pd.DataFrame,
    group_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratios_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratios_sum, 1.0, atol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0; got {ratios_sum}")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("All split ratios must be > 0")

    groups = df[group_col].astype(str).values
    all_idx = np.arange(len(df))

    first_split = GroupShuffleSplit(
        n_splits=1,
        train_size=train_ratio,
        test_size=(1.0 - train_ratio),
        random_state=seed,
    )
    train_idx, temp_idx = next(first_split.split(all_idx, groups=groups))

    temp_groups = groups[temp_idx]
    val_share_of_temp = val_ratio / (val_ratio + test_ratio)

    second_split = GroupShuffleSplit(
        n_splits=1,
        train_size=val_share_of_temp,
        test_size=(1.0 - val_share_of_temp),
        random_state=seed + 1,
    )
    val_rel_idx, test_rel_idx = next(second_split.split(temp_idx, groups=temp_groups))

    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def validate_splits(
    original_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
) -> dict[str, int]:
    train_genes = set(train_df[group_col].astype(str))
    val_genes = set(val_df[group_col].astype(str))
    test_genes = set(test_df[group_col].astype(str))

    train_val_overlap = len(train_genes & val_genes)
    train_test_overlap = len(train_genes & test_genes)
    val_test_overlap = len(val_genes & test_genes)

    if train_val_overlap != 0:
        raise RuntimeError(f"train/val overlap found: {train_val_overlap} genes")
    if train_test_overlap != 0:
        raise RuntimeError(f"train/test overlap found: {train_test_overlap} genes")
    if val_test_overlap != 0:
        raise RuntimeError(f"val/test overlap found: {val_test_overlap} genes")

    total_after = len(train_df) + len(val_df) + len(test_df)
    if total_after != len(original_df):
        raise RuntimeError(
            f"Row count mismatch after split: {total_after} vs original {len(original_df)}"
        )

    overall_ratio = float((original_df["label"] == 1).mean())

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_ratio = float((split_df["label"] == 1).mean()) if len(split_df) else 0.0
        if abs(split_ratio - overall_ratio) > 0.05:
            raise RuntimeError(
                f"Label ratio check failed for {split_name}: {split_ratio:.4f} "
                f"vs overall {overall_ratio:.4f} (difference > 0.05)"
            )

    return {
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
    }



def print_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    rows = []
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        pathogenic, benign, pathogenic_ratio = label_stats(df)
        genes = int(df["gene"].astype(str).nunique()) if "gene" in df.columns else int(df["GeneSymbol"].astype(str).nunique())
        rows.append((name, len(df), genes, pathogenic, benign, pathogenic_ratio))

    print("\nSplit summary")
    print("| Split | Rows | Genes | Pathogenic | Benign | Ratio |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, n_rows, n_genes, pathogenic, benign, ratio in rows:
        print(f"| {name} | {n_rows:,} | {n_genes:,} | {pathogenic:,} | {benign:,} | {ratio:.4f} |")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    input_path = resolve_path(repo_root, args.input)
    output_dir = resolve_path(repo_root, args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path)
    if "label" not in df.columns:
        raise ValueError("Input dataset must contain 'label' column")

    group_col = get_gene_column(df)
    print(f"Input: {input_path}")
    print(f"Rows: {len(df):,}, Columns: {len(df.columns):,}")
    print(f"Group column: {group_col}")

    gene_counts = df[group_col].astype(str).value_counts()
    print(f"Unique genes: {gene_counts.shape[0]:,}")

    train_df, val_df, test_df = split_dataframe(
        df=df,
        group_col=group_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    overlap = validate_splits(
        original_df=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        group_col=group_col,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print_summary(train_df, val_df, test_df)
    print("✅ Gene-level split validated — zero overlap between splits")
    print(f"Saved train: {train_path}")
    print(f"Saved val: {val_path}")
    print(f"Saved test: {test_path}")


if __name__ == "__main__":
    main()
