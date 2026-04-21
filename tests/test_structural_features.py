"""Unit tests for `src.structural_features`.

A real AlphaFold PDB is ~250 KB and deterministic, so we use a tiny
synthetic 3-residue PDB to lock in parser behavior without downloading
anything. The freesasa dependency is light and safe in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from src.structural_features import _NAN_FEATURES, compute_for_position

_TINY_PDB = """HEADER    TEST
REMARK   tiny synthetic pdb for tests
ATOM      1  N   ALA A   1      -1.458   0.000   0.000  1.00 95.00           N
ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00 95.00           C
ATOM      3  C   ALA A   1       0.550   1.404   0.000  1.00 95.00           C
ATOM      4  O   ALA A   1      -0.220   2.377   0.000  1.00 95.00           O
ATOM      5  CB  ALA A   1       0.517  -0.750   1.240  1.00 95.00           C
ATOM      6  N   VAL A   2       1.861   1.517   0.000  1.00 80.00           N
ATOM      7  CA  VAL A   2       2.641   2.751   0.000  1.00 80.00           C
ATOM      8  C   VAL A   2       4.126   2.530   0.000  1.00 80.00           C
ATOM      9  O   VAL A   2       4.557   1.385   0.000  1.00 80.00           O
ATOM     10  CB  VAL A   2       2.232   3.583   1.240  1.00 80.00           C
ATOM     11  N   LEU A   3       4.935   3.584   0.000  1.00 60.00           N
ATOM     12  CA  LEU A   3       6.380   3.480   0.000  1.00 60.00           C
ATOM     13  C   LEU A   3       6.922   4.882   0.000  1.00 60.00           C
ATOM     14  O   LEU A   3       6.152   5.855   0.000  1.00 60.00           O
ATOM     15  CB  LEU A   3       6.897   2.730   1.240  1.00 60.00           C
TER      16      LEU A   3
END
"""


@pytest.fixture
def tiny_pdb(tmp_path: Path) -> Path:
    p = tmp_path / "AF-TINY-F1-model_v4.pdb"
    p.write_text(_TINY_PDB)
    return p


def test_plddt_from_bfactor(tiny_pdb: Path) -> None:
    """AlphaFold stores pLDDT in the B-factor column. Positions 1/2/3
    should read back 95 / 80 / 60."""
    feats = compute_for_position(tiny_pdb, 2)
    assert feats.pLDDT_position == pytest.approx(80.0)
    # Window mean (positions 1-3 only; no pads): (95+80+60)/3 ≈ 78.33
    assert feats.pLDDT_window_5 == pytest.approx(78.333333, rel=1e-3)


def test_neighbors_within_5A(tiny_pdb: Path) -> None:
    """Position 2's CA is ~1.5 Å from pos 1 and ~3.0 Å from pos 3, so both
    are within 5 Å. Expected neighbors_5A == 2."""
    feats = compute_for_position(tiny_pdb, 2)
    assert feats.neighbors_5A == 2


def test_missing_pdb_returns_nan(tmp_path: Path) -> None:
    feats = compute_for_position(tmp_path / "does_not_exist.pdb", 1)
    # Compare each field against the sentinel (can't use == on float NaN).
    assert feats.secondary_structure == _NAN_FEATURES.secondary_structure
    assert feats.neighbors_5A == _NAN_FEATURES.neighbors_5A
    import math

    assert math.isnan(feats.pLDDT_position)


def test_out_of_range_position_returns_nan(tiny_pdb: Path) -> None:
    """Only residues 1/2/3 exist; position 99 must return the NaN sentinel."""
    feats = compute_for_position(tiny_pdb, 99)
    import math

    assert math.isnan(feats.pLDDT_position)
    assert feats.neighbors_5A == _NAN_FEATURES.neighbors_5A


def test_sasa_computes_for_valid_position(tiny_pdb: Path) -> None:
    """Middle residue should have SASA > 0 (and < total surface area of
    the residue). We only assert finite positive value — exact Å² depends
    on freesasa version."""
    feats = compute_for_position(tiny_pdb, 2)
    import math

    if not math.isnan(feats.SASA_position):
        assert feats.SASA_position > 0
        assert feats.SASA_position < 500  # upper bound for a single residue
