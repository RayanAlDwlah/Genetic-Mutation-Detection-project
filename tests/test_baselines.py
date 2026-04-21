"""Unit tests for the Stage-1 baseline comparison infrastructure.

We test `evaluate_baseline` end-to-end with a synthetic perfect
classifier and with a random score, so the metric math is locked in.
The AlphaMissense extractor is tested against a tiny synthetic TSV
that matches the real file's schema.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.baselines.alphamissense import (
    _canonical_key,
    build_lookup,
    extract_scores_for_keys,
)
from src.baselines.evaluate import BaselineMetadata, evaluate_baseline

# ──────────────────────── evaluate_baseline ────────────────────────


def _make_test_df(n: int, positive_rate: float, rng: np.random.Generator) -> pd.DataFrame:
    labels = (rng.uniform(size=n) < positive_rate).astype(int)
    return pd.DataFrame(
        {
            "variant_key": [f"1:{i}:A:G" for i in range(n)],
            "gene": ["GENE_A"] * (n // 2) + ["GENE_B"] * (n - n // 2),
            "label": labels,
        }
    )


class TestEvaluateBaseline:
    def test_perfect_scorer_gets_auc_1(self, rng: np.random.Generator) -> None:
        df = _make_test_df(200, 0.3, rng)
        score = df["label"].astype(float)  # perfect signal
        meta = BaselineMetadata(
            name="perfect",
            display_name="Perfect",
            year=2026,
            training_data="n/a",
        )
        out = evaluate_baseline(meta=meta, test_df=df, test_score=score, n_boot=50)
        row = out.iloc[0]
        assert row["slice"] == "clinvar_test"
        assert row["roc_auc"] == pytest.approx(1.0)
        assert row["pr_auc"] == pytest.approx(1.0)

    def test_metadata_columns_propagate(self, rng: np.random.Generator) -> None:
        df = _make_test_df(100, 0.4, rng)
        meta = BaselineMetadata(
            name="toy",
            display_name="Toy Scorer",
            year=1999,
            training_data="test_data",
            training_contamination_warning="yes",
        )
        out = evaluate_baseline(
            meta=meta,
            test_df=df,
            test_score=df["label"].astype(float),
            n_boot=50,
        )
        assert out["baseline"].iloc[0] == "toy"
        assert out["baseline_display_name"].iloc[0] == "Toy Scorer"
        assert out["year"].iloc[0] == 1999
        assert out["training_contamination_warning"].iloc[0] == "yes"

    def test_higher_is_pathogenic_false_flips_sign(self, rng: np.random.Generator) -> None:
        """SIFT is 'lower = damaging'; passing `higher_is_pathogenic=False`
        must invert the score so the perfect inverted scorer still gets
        AUC=1."""
        df = _make_test_df(200, 0.3, rng)
        score = 1.0 - df["label"].astype(float)  # inverted: 0 = damaging
        meta = BaselineMetadata(
            name="inverted",
            display_name="Inv",
            year=2026,
            training_data="n/a",
            higher_is_pathogenic=False,
        )
        out = evaluate_baseline(meta=meta, test_df=df, test_score=score, n_boot=50)
        assert out["roc_auc"].iloc[0] == pytest.approx(1.0)

    def test_nan_scores_counted_as_uncovered(self, rng: np.random.Generator) -> None:
        df = _make_test_df(200, 0.3, rng)
        score = pd.Series([np.nan] * 50 + df["label"].astype(float).tolist()[50:])
        meta = BaselineMetadata(
            name="half",
            display_name="Half",
            year=2026,
            training_data="n/a",
        )
        out = evaluate_baseline(meta=meta, test_df=df, test_score=score, n_boot=50)
        assert out["coverage"].iloc[0] == pytest.approx(150 / 200)
        assert out["n"].iloc[0] == 150


# ──────────────────────── AlphaMissense extractor ────────────────────────


class TestAlphaMissenseExtractor:
    @pytest.fixture
    def tiny_am_tsv(self, tmp_path: Path) -> Path:
        """5-row AlphaMissense-style gzipped TSV."""
        p = tmp_path / "AlphaMissense_tiny.tsv.gz"
        header = (
            "# copyright\n"
            "# license\n"
            "#CHROM\tPOS\tREF\tALT\tgenome\tuniprot_id\ttranscript_id\t"
            "protein_variant\tam_pathogenicity\tam_class\n"
        )
        rows = [
            "chr1\t100\tA\tG\thg19\tQ1\tENST1\tM1V\t0.15\tbenign\n",
            "chr1\t200\tC\tT\thg19\tQ2\tENST2\tP2L\t0.85\tpathogenic\n",
            "chr17\t41244936\tG\tA\thg19\tQ3\tENST3\tP575L\t0.90\tpathogenic\n",
            "chr7\t117199644\tC\tT\thg19\tQ4\tENST4\tI507F\t0.98\tpathogenic\n",
            "chrX\t500\tA\tG\thg19\tQ5\tENST5\tT1A\t0.30\tambiguous\n",
        ]
        with gzip.open(p, "wt") as fh:
            fh.write(header)
            fh.writelines(rows)
        return p

    def test_canonical_key_strips_chr_prefix(self) -> None:
        assert _canonical_key("chr17", 41244936, "g", "a") == "17:41244936:G:A"
        assert _canonical_key("M", 100, "A", "G") == "MT:100:A:G"
        assert _canonical_key("chrMT", 100, "A", "G") == "MT:100:A:G"

    def test_extract_subset(self, tiny_am_tsv: Path) -> None:
        keys = {"17:41244936:G:A", "7:117199644:C:T", "99:999:A:G"}  # last one absent
        out = extract_scores_for_keys(tiny_am_tsv, keys, progress=False)
        assert len(out) == 2
        assert set(out["variant_key"]) == {"17:41244936:G:A", "7:117199644:C:T"}
        brca = out[out["variant_key"] == "17:41244936:G:A"].iloc[0]
        assert brca["am_pathogenicity"] == pytest.approx(0.90)
        assert brca["am_class"] == "pathogenic"

    def test_build_lookup_caches_to_parquet(self, tiny_am_tsv: Path, tmp_path: Path) -> None:
        cache = tmp_path / "am_cache.parquet"
        query = pd.DataFrame({"variant_key": ["1:100:A:G", "1:200:C:T", "NOT:FOUND:X:Y"]})
        r1 = build_lookup(tsv_gz_path=tiny_am_tsv, query_df=query, cache_path=cache, progress=False)
        assert cache.exists()
        assert len(r1.scores) == 2
        # Second call hits the cache without re-scanning.
        r2 = build_lookup(tsv_gz_path=tiny_am_tsv, query_df=query, cache_path=cache, progress=False)
        assert len(r2.scores) == 2
        assert r1.scores.equals(r2.scores)
