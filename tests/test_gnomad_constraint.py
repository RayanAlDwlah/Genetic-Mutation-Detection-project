"""Unit tests for `src.gnomad_constraint.merge_constraint`.

This module implements the "train-only median" imputation pattern that we
reuse for every subsequent feature (ESM-2 LLR, structural features). A
bug here = silent leakage. We lock in the invariant explicitly: medians
fit on val/test must never influence imputation of test rows.
"""

from __future__ import annotations

import gzip

import pandas as pd
import pytest
from src.gnomad_constraint import CONSTRAINT_COLS, load_constraint_table, merge_constraint


@pytest.fixture
def constraint_table() -> pd.DataFrame:
    """Minimal in-memory gnomAD-style constraint table: 3 genes with
    distinct numbers so we can verify which value landed on which row."""
    return pd.DataFrame(
        {
            "gene": ["GENE_A", "GENE_B", "GENE_C"],
            "pLI": [0.90, 0.10, 0.50],
            "oe_lof_upper": [0.20, 0.80, 0.50],
            "mis_z": [3.0, 0.1, 1.5],
            "oe_mis_upper": [0.30, 0.95, 0.60],
            "lof_z": [5.0, 0.0, 2.5],
        }
    )


@pytest.fixture
def variants_frame() -> pd.DataFrame:
    """4 rows, 3 different genes, one MISSING gene to exercise imputation."""
    return pd.DataFrame(
        {
            "gene": ["GENE_A", "GENE_B", "GENE_C", "GENE_UNKNOWN"],
            "label": [1, 0, 1, 0],
        }
    )


class TestMergeConstraint:
    def test_all_constraint_cols_added(
        self, variants_frame: pd.DataFrame, constraint_table: pd.DataFrame
    ) -> None:
        merged, _ = merge_constraint(variants_frame, constraint=constraint_table)
        for col in CONSTRAINT_COLS:
            assert col in merged.columns

    def test_imputation_flag_matches_gene_presence(
        self, variants_frame: pd.DataFrame, constraint_table: pd.DataFrame
    ) -> None:
        merged, _ = merge_constraint(variants_frame, constraint=constraint_table)
        # 3 known genes → flag=0, 1 unknown gene → flag=1
        assert (
            merged.loc[merged["gene"] == "GENE_UNKNOWN", "is_imputed_gnomad_constraint"].iloc[0]
            == 1
        )
        for gene in ("GENE_A", "GENE_B", "GENE_C"):
            assert merged.loc[merged["gene"] == gene, "is_imputed_gnomad_constraint"].iloc[0] == 0

    def test_values_correctly_joined(
        self, variants_frame: pd.DataFrame, constraint_table: pd.DataFrame
    ) -> None:
        merged, _ = merge_constraint(variants_frame, constraint=constraint_table)
        row_a = merged.loc[merged["gene"] == "GENE_A"].iloc[0]
        assert row_a["pLI"] == pytest.approx(0.90)
        assert row_a["oe_lof_upper"] == pytest.approx(0.20)

    def test_train_only_medians_computed_correctly(
        self, variants_frame: pd.DataFrame, constraint_table: pd.DataFrame
    ) -> None:
        """When no `impute_medians` is passed, medians come from the 3 rows
        where the gene was present — NOT including the imputed row."""
        merged, medians = merge_constraint(variants_frame, constraint=constraint_table)
        # Median of [0.90, 0.10, 0.50] = 0.50
        assert medians["pLI"] == pytest.approx(0.50)
        # Median of [0.20, 0.80, 0.50] = 0.50
        assert medians["oe_lof_upper"] == pytest.approx(0.50)

    def test_passed_medians_applied_verbatim(
        self, variants_frame: pd.DataFrame, constraint_table: pd.DataFrame
    ) -> None:
        """When `impute_medians` is passed (e.g. the train-fit dict reused
        for val/test), they must be applied to missing rows unchanged."""
        train_medians = dict.fromkeys(CONSTRAINT_COLS, 99.0)
        merged, returned = merge_constraint(
            variants_frame, constraint=constraint_table, impute_medians=train_medians
        )
        # Same dict handed back
        assert returned == train_medians
        # Unknown gene gets 99 for every constraint column
        row_unknown = merged.loc[merged["gene"] == "GENE_UNKNOWN"].iloc[0]
        for col in CONSTRAINT_COLS:
            assert row_unknown[col] == pytest.approx(99.0)

    def test_no_leakage_when_using_train_medians_on_val(
        self, constraint_table: pd.DataFrame
    ) -> None:
        """Regression: simulate train/val pipeline. The val merge must use
        train's medians, not its own."""
        train = pd.DataFrame({"gene": ["GENE_A", "GENE_B"], "label": [1, 0]})
        val = pd.DataFrame({"gene": ["GENE_UNKNOWN_V"], "label": [0]})

        _, train_medians = merge_constraint(train, constraint=constraint_table)
        val_merged, returned_medians = merge_constraint(
            val, constraint=constraint_table, impute_medians=train_medians
        )
        # Returned medians on val == train medians (not refit from val).
        assert returned_medians == train_medians
        # The imputed value for GENE_UNKNOWN_V must equal train's median,
        # NOT val's median (val has only one row, which would be degenerate).
        train_pLI_median = train_medians["pLI"]
        assert val_merged["pLI"].iloc[0] == pytest.approx(train_pLI_median)

    def test_missing_gene_column_raises(self, constraint_table: pd.DataFrame) -> None:
        bad = pd.DataFrame({"other_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="must have a `gene` column"):
            merge_constraint(bad, constraint=constraint_table)


class TestLoadConstraintTable:
    """`load_constraint_table` reads gnomAD's gzipped TSV, keeps the columns
    we care about, and deduplicates by gene (keeping the most-constrained
    row per gene)."""

    def test_roundtrip_and_dedup(self, tmp_path) -> None:
        path = tmp_path / "lof_metrics.txt.bgz"
        # Two rows for GENE_A — lower oe_lof_upper should win (more constrained).
        table = pd.DataFrame(
            {
                "gene": ["GENE_A", "GENE_A", "GENE_B"],
                "pLI": [0.1, 0.9, 0.5],
                "oe_lof_upper": [0.8, 0.2, 0.5],
                "mis_z": [0.1, 3.0, 1.0],
                "oe_mis_upper": [0.9, 0.3, 0.6],
                "lof_z": [0.0, 5.0, 2.0],
                "extra_col_we_dont_want": [1, 2, 3],
            }
        )
        with gzip.open(path, "wt") as fh:
            table.to_csv(fh, sep="\t", index=False)

        loaded = load_constraint_table(path)

        # Extra columns dropped.
        assert set(loaded.columns) == {"gene", *CONSTRAINT_COLS}
        # Deduplicated to one row per gene.
        assert len(loaded) == 2
        assert sorted(loaded["gene"].tolist()) == ["GENE_A", "GENE_B"]
        # Most-constrained GENE_A row wins (pLI=0.9, oe_lof_upper=0.2).
        row_a = loaded[loaded["gene"] == "GENE_A"].iloc[0]
        assert row_a["pLI"] == pytest.approx(0.9)
        assert row_a["oe_lof_upper"] == pytest.approx(0.2)
