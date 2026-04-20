"""Unit + regression tests for `src.data_splitting.assign_gene_family`.

The paralog-aware family grouper is the *single source of truth* that keeps
train/val/test disjoint at the gene-family level. Any change to the regex
list in `_FAMILY_PATTERNS` must be covered by explicit test cases; otherwise
a future "cleanup" could silently reintroduce paralog leakage.

This file also regression-tests the committed splits: they must remain
family-disjoint bit-for-bit.
"""

from __future__ import annotations

import pandas as pd
import pytest
from src.data_splitting import assign_gene_family


class TestAssignGeneFamily:
    """Known-family collapses — every row is a locked-in expectation."""

    @pytest.mark.parametrize(
        ("gene", "family"),
        [
            # Keratin-associated proteins
            ("KRTAP1-1", "KRTAP"),
            ("KRTAP10-11", "KRTAP"),
            # Keratins (numbered) — KRT1 and KRT18 must share a family
            ("KRT1", "KRT"),
            ("KRT18", "KRT"),
            ("KRT85", "KRT"),
            # HLA genes
            ("HLA-A", "HLA"),
            ("HLA-DRB5", "HLA"),
            # Zinc fingers — large paralog cluster
            ("ZNF10", "ZNF"),
            ("ZNF804A", "ZNF"),
            # Solute carriers — family preserved (SLC6, SLC17, …)
            ("SLC6A4", "SLC6"),
            ("SLC17A3", "SLC17"),
            # Cadherins — numbered
            ("CDH1", "CDH"),
            ("CDH23", "CDH"),
            # Protocadherins — any PCDH prefix
            ("PCDH15", "PCDH"),
            ("PCDHGA3", "PCDH"),
            # TRIM, TMEM, CCDC, LRRC, ANKR — all large paralog clusters
            ("TRIM5", "TRIM"),
            ("TRIM67", "TRIM"),
            ("TMEM123", "TMEM"),
            ("CCDC22", "CCDC"),
            ("LRRC32", "LRRC"),
            ("ANKRD11", "ANKR"),
            # Olfactory receptors — subfamily preserved (OR1, OR2, …)
            ("OR1A1", "OR1"),
            ("OR51E2", "OR51"),
            # Ribosomal proteins
            ("RPL5", "RPL"),
            ("RPS6", "RPS"),
            # Mitochondrial genes
            ("MT-ND4", "MT"),
            ("MT-CO1", "MT"),
        ],
    )
    def test_known_family_assignments(self, gene: str, family: str) -> None:
        assert assign_gene_family(gene) == family

    def test_numeric_suffix_fallback_collapses_foxa(self) -> None:
        """Genes not covered by the regex list fall back to stripping
        trailing digits: FOXA1/FOXA2/FOXA3 → FOXA."""
        assert (
            assign_gene_family("FOXA1")
            == assign_gene_family("FOXA2")
            == assign_gene_family("FOXA3")
            == "FOXA"
        )

    def test_unrelated_genes_get_different_families(self) -> None:
        """Smoke test the core separation property — MYO5A and CDH1 must
        never collapse together."""
        assert assign_gene_family("MYO5A") != assign_gene_family("CDH1")
        assert assign_gene_family("TP53") != assign_gene_family("BRCA1")

    def test_none_input_returns_empty_string(self) -> None:
        assert assign_gene_family(None) == ""

    def test_case_insensitive(self) -> None:
        """Lowercase gene names must still resolve."""
        assert assign_gene_family("krt1") == "KRT"
        assert assign_gene_family("znf10") == "ZNF"

    def test_uncovered_gene_returns_stripped_symbol(self) -> None:
        """A single-gene family (no paralogs, no trailing digits) falls
        through to the input gene symbol itself."""
        assert assign_gene_family("APOE") == "APOE"


class TestCommittedSplitsAreFamilyDisjoint:
    """Regression: the committed train/val/test must have zero shared
    gene families. If this test fails, the splitter has been tampered
    with in a way that reintroduces paralog leakage."""

    def test_zero_family_overlap_train_test(
        self, train_split: pd.DataFrame, test_split: pd.DataFrame
    ) -> None:
        train_fams = {assign_gene_family(g) for g in train_split["gene"].unique()}
        test_fams = {assign_gene_family(g) for g in test_split["gene"].unique()}
        overlap = train_fams & test_fams
        assert overlap == set(), (
            f"paralog leakage: {len(overlap)} families in both train and test "
            f"(sample: {sorted(overlap)[:5]})"
        )

    def test_zero_family_overlap_train_val(
        self, train_split: pd.DataFrame, val_split: pd.DataFrame
    ) -> None:
        train_fams = {assign_gene_family(g) for g in train_split["gene"].unique()}
        val_fams = {assign_gene_family(g) for g in val_split["gene"].unique()}
        assert not (train_fams & val_fams)

    def test_zero_family_overlap_val_test(
        self, val_split: pd.DataFrame, test_split: pd.DataFrame
    ) -> None:
        val_fams = {assign_gene_family(g) for g in val_split["gene"].unique()}
        test_fams = {assign_gene_family(g) for g in test_split["gene"].unique()}
        assert not (val_fams & test_fams)

    def test_label_balance_across_splits(
        self,
        train_split: pd.DataFrame,
        val_split: pd.DataFrame,
        test_split: pd.DataFrame,
    ) -> None:
        """Stratification sanity: pathogenic/benign ratio must be within
        ±8 pp across splits."""
        rates = {
            "train": train_split["label"].mean(),
            "val": val_split["label"].mean(),
            "test": test_split["label"].mean(),
        }
        gap = max(rates.values()) - min(rates.values())
        assert gap < 0.08, f"label-rate gap {gap:.3f} > 0.08: {rates}"
