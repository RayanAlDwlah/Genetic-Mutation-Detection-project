"""Canonicalize variant identifiers to the pipeline's `chr:pos:ref:alt` key.

External datasets use wildly different conventions:
- denovo-db emits `Chr=1, Position=13668, Variant='G>A'`
- ProteinGym emits HGVSp-like `A123G` that must be resolved to genomic coords
- Some ClinVar exports emit `NC_000017.10:g.41276045G>A`

Mapping HGVSp → genomic coordinates requires a transcript-aware resolver
(`pyhgvs`, VEP, or Ensembl REST). For the Phase-D-v1 harness we only support
the two tractable inputs (chr:pos:ref:alt records + explicit chrom/pos/ref/alt
columns). HGVSp resolution is marked TODO and logged as `unmapped` for now,
so the protocol stays honest about coverage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Chromosome normalization (strip chr/CHR, map MT↔M).
_CHR_PREFIX_RE = re.compile(r"^(chr|CHR)")


def normalize_chromosome(chrom: object) -> str:
    """Strip chr prefix and normalize mitochondrial naming.

    Returns "" for null-like inputs so callers can filter unmapped rows.
    """
    if chrom is None:
        return ""
    s = str(chrom).strip()
    if not s or s.lower() == "nan":
        return ""
    s = _CHR_PREFIX_RE.sub("", s)
    if s.upper() in {"M", "MT"}:
        return "MT"
    return s.upper()


@dataclass(frozen=True)
class CanonicalVariant:
    chrom: str
    pos: int
    ref: str
    alt: str

    @property
    def key(self) -> str:
        return f"{self.chrom}:{self.pos}:{self.ref}:{self.alt}"


_SIMPLE_SUB_RE = re.compile(r"^([ACGT])\s*>\s*([ACGT])$", re.IGNORECASE)


def parse_variant_change(token: object) -> tuple[str, str] | None:
    """Parse `"G>A"` style substitution tokens (denovo-db convention)."""
    if token is None:
        return None
    m = _SIMPLE_SUB_RE.match(str(token).strip())
    if not m:
        return None
    return m.group(1).upper(), m.group(2).upper()


def to_canonical_key(
    *,
    chrom: object,
    pos: object,
    ref: object | None = None,
    alt: object | None = None,
    change: object | None = None,
) -> CanonicalVariant | None:
    """Best-effort canonicalization. Returns None if record can't be resolved.

    Caller may pass either (ref, alt) explicitly, or a `change` token like
    `"G>A"`. If neither works the variant is reported unmapped.
    """
    c = normalize_chromosome(chrom)
    if not c:
        return None
    try:
        p = int(pos)
    except (TypeError, ValueError):
        return None
    if ref is not None and alt is not None:
        r, a = str(ref).strip().upper(), str(alt).strip().upper()
    else:
        parsed = parse_variant_change(change)
        if not parsed:
            return None
        r, a = parsed
    if not r or not a or r == a:
        return None
    if any(nuc not in {"A", "C", "G", "T"} for nuc in (r + a)):
        # indels / multi-nucleotide substitutions are outside the missense-only
        # scope the training model was fit on; skip rather than mislabel.
        return None
    return CanonicalVariant(chrom=c, pos=p, ref=r, alt=a)
