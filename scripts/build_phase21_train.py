#!/usr/bin/env python3
# Added for Phase 2.1 (S2): merge ESM-2 LLR into the canonical splits.
"""Build Phase-2.1 split parquets that add esm2_llr + is_imputed_esm2_llr.

Idempotent. Never mutates the canonical `data/splits/{train,val,test}.parquet`
files. Writes to `data/splits/phase21/` (gitignore-exempted in a sibling commit).

Aggregation rule: per-variant `min(esm2_llr)` (most-pathogenic isoform).
S1 verified that the score parquets currently have one row per variant_key
on test split; the rule still applies cleanly because pandas groupby on a
unique key reduces to identity.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def load_split(split: str) -> pd.DataFrame:
    return pd.read_parquet(REPO / f"data/splits/{split}.parquet")


def load_scores(split: str) -> pd.DataFrame:
    return pd.read_parquet(REPO / f"data/intermediate/esm2/scores_{split}.parquet")


def aggregate_min(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per variant_key with min(esm2_llr) over isoforms.

    Drops rows where esm2_llr is NaN before aggregation so that "skipped"
    isoforms do not pollute the min. If every isoform of a variant is
    skipped, the variant is absent from the aggregated frame and will be
    treated as imputed downstream.
    """
    valid = scores.dropna(subset=["esm2_llr"]).copy()
    agg = valid.groupby("variant_key", as_index=False)["esm2_llr"].min()
    return agg


def merge_split(split_df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    out = split_df.merge(agg, on="variant_key", how="left")
    out["is_imputed_esm2_llr"] = out["esm2_llr"].isna().astype(int)
    return out


def coverage_report(name: str, df: pd.DataFrame) -> None:
    n = len(df)
    n_present = int((df["is_imputed_esm2_llr"] == 0).sum())
    n_imputed = int((df["is_imputed_esm2_llr"] == 1).sum())
    by_class = df.groupby("label")["esm2_llr"].agg(["mean", "median", "count"])
    print(f"[build_phase21] {name}: n={n}  esm2 present={n_present} ({n_present/n:.2%})  "
          f"imputed={n_imputed} ({n_imputed/n:.2%})")
    print(f"  per-class LLR (lower = more pathogenic):")
    for label, row in by_class.iterrows():
        print(f"    label={int(label)}: mean={row['mean']:.4f}  median={row['median']:.4f}  n={int(row['count'])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO / "data/splits/phase21")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        sdf = load_split(split)
        scores = load_scores(split)
        agg = aggregate_min(scores)
        merged = merge_split(sdf, agg)
        out = args.out_dir / f"{split}.parquet"
        merged.to_parquet(out, index=False)
        print(f"[build_phase21] wrote {out}  shape={merged.shape}  cols={len(merged.columns)}")
        coverage_report(split, merged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
