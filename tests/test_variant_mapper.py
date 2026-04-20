"""Unit tests for `src.external_validation.variant_mapper`.

The canonical-key builder is the single point of contact between external
datasets (denovo-db, ProteinGym, literature-curated) and the training
feature space. Subtle bugs here cause silent drops in coverage — these
tests lock in the canonicalization contract.
"""

from __future__ import annotations

import pytest
from src.external_validation.variant_mapper import (
    CanonicalVariant,
    normalize_chromosome,
    parse_variant_change,
    to_canonical_key,
)


class TestNormalizeChromosome:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1", "1"),
            ("chr1", "1"),
            ("CHR17", "17"),
            ("X", "X"),
            ("chrY", "Y"),
            ("M", "MT"),
            ("MT", "MT"),
            ("chrMT", "MT"),
        ],
    )
    def test_known(self, raw: str, expected: str) -> None:
        assert normalize_chromosome(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "  ", "nan", "NaN"])
    def test_null_like(self, raw: object) -> None:
        assert normalize_chromosome(raw) == ""


class TestParseVariantChange:
    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("G>A", ("G", "A")),
            ("c>t", ("C", "T")),  # lowercase normalizes
            ("A > G", ("A", "G")),
        ],
    )
    def test_valid(self, token: str, expected: tuple[str, str]) -> None:
        assert parse_variant_change(token) == expected

    @pytest.mark.parametrize("token", [None, "", "A/G", "GGA>A", "X>Y", "ATCG"])
    def test_invalid_returns_none(self, token: object) -> None:
        assert parse_variant_change(token) is None


class TestToCanonicalKey:
    def test_simple_explicit_ref_alt(self) -> None:
        v = to_canonical_key(chrom="chr1", pos="12345", ref="A", alt="G")
        assert v is not None
        assert v.chrom == "1" and v.pos == 12345
        assert v.ref == "A" and v.alt == "G"
        assert v.key == "1:12345:A:G"

    def test_parse_change_token_when_ref_alt_missing(self) -> None:
        v = to_canonical_key(chrom="17", pos=41244936, change="G>A")
        assert v is not None
        assert v.key == "17:41244936:G:A"

    def test_same_ref_and_alt_rejected(self) -> None:
        """Not a real variant."""
        assert to_canonical_key(chrom="1", pos=100, ref="A", alt="A") is None

    def test_non_acgt_rejected(self) -> None:
        """Indels / multi-nucleotide substitutions aren't missense."""
        assert to_canonical_key(chrom="1", pos=100, ref="AT", alt="G") is None
        assert to_canonical_key(chrom="1", pos=100, ref="N", alt="G") is None

    def test_unparsable_position_rejected(self) -> None:
        assert to_canonical_key(chrom="1", pos="not_a_number", ref="A", alt="G") is None

    def test_null_chrom_rejected(self) -> None:
        assert to_canonical_key(chrom=None, pos=100, ref="A", alt="G") is None

    def test_returns_frozen_dataclass(self) -> None:
        v = to_canonical_key(chrom="1", pos=100, ref="A", alt="G")
        assert isinstance(v, CanonicalVariant)
        # Frozen dataclass: attribute assignment raises.
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            v.pos = 999  # type: ignore[misc]
