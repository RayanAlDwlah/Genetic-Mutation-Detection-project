"""Unit tests for `src.external_validation.denovo_loader.load_denovo_db`.

The loader has three important jobs:
  1. Keep only `FunctionClass == missense*` rows.
  2. Label = 1 for pathogenic phenotypes, 0 for documented controls;
     drop anything else (no silent benign).
  3. Canonicalize chr:pos:ref:alt via variant_mapper.

These tests synthesize a tiny TSV and assert each policy.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from src.external_validation.denovo_loader import load_denovo_db


@pytest.fixture
def tiny_denovo_tsv(tmp_path: Path) -> Path:
    """Write a 5-row denovo-db-style TSV covering all key cases."""
    header = [
        "#SampleID",
        "StudyName",
        "PubmedID",
        "NumProbands",
        "NumControls",
        "SequenceType",
        "PrimaryPhenotype",
        "Validation",
        "Chr",
        "Position",
        "Variant",
        "rsID",
        "FunctionClass",
        "cDnaVariant",
        "ProteinVariant",
        "Exon/Intron",
        "PolyPhen(HDiv)",
        "PolyPhen(HVar)",
        "SiftScore",
        "CaddScore",
        "LofScore",
        "LrtScore",
        "Gene",
        "Transcript",
    ]
    rows = [
        # 1. missense, affected → label=1
        [
            "S1",
            "Study1",
            "12345",
            "100",
            "0",
            "wholeExome",
            "autism",
            "yes",
            "17",
            "41244936",
            "G>A",
            ".",
            "missense",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            "BRCA1",
            "NM_007294",
        ],
        # 2. missense, sibling control → label=0
        [
            "S2",
            "Study2",
            "12345",
            "0",
            "50",
            "wholeExome",
            "siblingcontrol",
            "yes",
            "7",
            "117199644",
            "C>T",
            ".",
            "missense",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            "CFTR",
            "NM_000492",
        ],
        # 3. synonymous (non-missense) → DROPPED
        [
            "S3",
            "Study3",
            "12345",
            "100",
            "0",
            "wholeExome",
            "autism",
            "yes",
            "1",
            "100",
            "A>G",
            ".",
            "synonymous",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            "GENE3",
            "NM_100",
        ],
        # 4. missense with ambiguous phenotype → DROPPED
        [
            "S4",
            "Study4",
            "12345",
            "100",
            "0",
            "wholeExome",
            "unknown",
            "yes",
            "1",
            "200",
            "A>G",
            ".",
            "missense",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            "GENE4",
            "NM_200",
        ],
        # 5. missense, DD → label=1 (alternate pathogenic phenotype spelling)
        [
            "S5",
            "Study5",
            "12345",
            "100",
            "0",
            "wholeExome",
            "developmentalDisorder",
            "yes",
            "X",
            "300",
            "C>T",
            ".",
            "missense-near-splice",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            ".",
            "GENE5",
            "NM_300",
        ],
    ]
    p = tmp_path / "denovo.tsv"
    with p.open("w") as fh:
        # simulate denovo-db's optional ##version line
        fh.write("##version=test\n")
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    return p


def test_drops_non_missense(tiny_denovo_tsv: Path) -> None:
    df = load_denovo_db(tiny_denovo_tsv)
    # synonymous row (S3) was dropped.
    assert "GENE3" not in df["gene"].tolist()


def test_drops_unlabeled_phenotype(tiny_denovo_tsv: Path) -> None:
    df = load_denovo_db(tiny_denovo_tsv)
    # S4 had phenotype="unknown" → must not appear.
    assert "GENE4" not in df["gene"].tolist()


def test_labels_pathogenic_and_control(tiny_denovo_tsv: Path) -> None:
    df = load_denovo_db(tiny_denovo_tsv)
    brca1 = df[df["gene"] == "BRCA1"].iloc[0]
    cftr = df[df["gene"] == "CFTR"].iloc[0]
    gene5 = df[df["gene"] == "GENE5"].iloc[0]
    assert brca1["label"] == 1
    assert cftr["label"] == 0
    assert gene5["label"] == 1  # missense-near-splice + DD = pathogenic


def test_canonical_variant_key_format(tiny_denovo_tsv: Path) -> None:
    df = load_denovo_db(tiny_denovo_tsv)
    brca1 = df[df["gene"] == "BRCA1"].iloc[0]
    assert brca1["variant_key"] == "17:41244936:G:A"
    assert brca1["chr"] == "17"
    assert int(brca1["pos"]) == 41244936
    assert brca1["ref"] == "G"
    assert brca1["alt"] == "A"


def test_final_schema(tiny_denovo_tsv: Path) -> None:
    df = load_denovo_db(tiny_denovo_tsv)
    required = {"variant_key", "chr", "pos", "ref", "alt", "gene", "label", "study", "phenotype"}
    assert required.issubset(df.columns)


def test_deduplicates_by_variant_key(tmp_path: Path, tiny_denovo_tsv: Path) -> None:
    """If the same variant is reported twice, keep only one row."""
    # Append a duplicate of the BRCA1 row to the existing fixture.
    lines = tiny_denovo_tsv.read_text().splitlines()
    brca1_line = next(ln for ln in lines if "BRCA1" in ln)
    dup = tmp_path / "denovo_dup.tsv"
    dup.write_text("\n".join(lines + [brca1_line]) + "\n")
    df = load_denovo_db(dup)
    assert (df["variant_key"] == "17:41244936:G:A").sum() == 1
