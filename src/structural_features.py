"""Extract AlphaFold-based structural features per residue.

For every `(uniprot_id, protein_position)` we compute:

- `pLDDT_position`       AlphaFold per-residue confidence (0–100; >70 = confident)
- `pLDDT_window_5`       mean pLDDT over ±5-residue window (smooths edge noise)
- `SASA_position`        solvent-accessible surface area in Å² (freesasa)
- `secondary_structure`  DSSP 1-letter code (H, E, T, S, G, B, C, I)
- `neighbors_5A`         number of Cα atoms within 5 Å of the target residue

Scores for residues that fail extraction (invalid PDB, missing DSSP, out of
range position) are returned as NaN so the caller can apply the same
train-only-median imputation pattern used everywhere else in the pipeline.

Inputs
------
- PDB file path: `AF-{uniprot}-F1-model_v4.pdb` (downloaded from
  `alphafold.ebi.ac.uk/files/…`).
- Protein positions: 1-indexed integers.

Design notes
------------
This module is structured so it runs identically on the Colab GPU
notebook (see `notebooks/12_alphafold_features_colab.ipynb`) and in the
Docker container on the host. The Colab download and compute path is
the same as the local path; only the PDB cache location changes.

The module imports Biopython, freesasa, and (optionally) DSSP. All three
live in the Docker image and are added to the Colab env in the notebook.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StructuralFeatures:
    pLDDT_position: float
    pLDDT_window_5: float
    SASA_position: float
    secondary_structure: str
    neighbors_5A: int

    def as_row(self, uniprot: str, position: int) -> dict[str, object]:
        return {
            "uniprot_id": uniprot,
            "protein_position": position,
            **self.__dict__,
        }


_NAN_FEATURES = StructuralFeatures(
    pLDDT_position=float("nan"),
    pLDDT_window_5=float("nan"),
    SASA_position=float("nan"),
    secondary_structure="",
    neighbors_5A=-1,
)


def _parse_pdb_residues(pdb_path: Path):
    """Parse a PDB file and return a Biopython structure with all CA atoms
    and their B-factor (which AlphaFold uses to encode pLDDT)."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    return structure


def _plddt_series(structure) -> dict[int, float]:
    """Extract pLDDT from CA atom B-factor per residue. AlphaFold PDBs
    embed pLDDT in the B-factor column, so the mapping is trivial."""
    out: dict[int, float] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":  # skip HETATM
                    continue
                try:
                    ca = residue["CA"]
                    out[residue.id[1]] = float(ca.get_bfactor())
                except KeyError:
                    continue
    return out


def _sasa_series(pdb_path: Path) -> dict[int, float]:
    """Per-residue SASA (Å²) via freesasa.

    Uses `freesasa.calc(structure)` (function, not class — the 2.x API).
    """
    import freesasa

    try:
        structure = freesasa.Structure(str(pdb_path))
        result = freesasa.calc(structure)
        residue_areas = result.residueAreas()
    except Exception:
        return {}

    out: dict[int, float] = {}
    for _chain, residues in residue_areas.items():
        for res_num_str, area in residues.items():
            try:
                res_num = int(res_num_str)
                out[res_num] = float(area.total)
            except (ValueError, AttributeError):
                continue
    return out


def _ss_series(pdb_path: Path) -> dict[int, str]:
    """Per-residue DSSP 1-letter secondary structure code."""
    try:
        from Bio.PDB import DSSP, PDBParser
    except ImportError:
        return {}
    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    try:
        dssp = DSSP(structure[0], str(pdb_path), dssp="dssp")
    except Exception:  # DSSP binary missing → non-fatal
        return {}
    out: dict[int, str] = {}
    for (_chain, res_id), record in dssp.property_dict.items():
        if not isinstance(res_id, tuple) or res_id[0] != " ":
            continue
        out[res_id[1]] = str(record[2])  # DSSP code
    return out


def _ca_coords(structure) -> dict[int, np.ndarray]:
    """Collect CA Cartesian coordinates per residue for neighbor counting."""
    out: dict[int, np.ndarray] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                try:
                    out[residue.id[1]] = np.asarray(residue["CA"].coord, dtype=float)
                except KeyError:
                    continue
    return out


def compute_for_position(pdb_path: Path, position: int) -> StructuralFeatures:
    """Return a `StructuralFeatures` tuple for one (pdb, position). Any
    error → all-NaN sentinel."""
    if not pdb_path.exists():
        return _NAN_FEATURES

    try:
        structure = _parse_pdb_residues(pdb_path)
    except Exception:
        return _NAN_FEATURES

    plddt = _plddt_series(structure)
    sasa = _sasa_series(pdb_path)
    ss = _ss_series(pdb_path)
    coords = _ca_coords(structure)

    if position not in plddt:
        return _NAN_FEATURES

    # Window pLDDT (±5).
    window_vals = [plddt[p] for p in range(position - 5, position + 6) if p in plddt]
    window_mean = float(np.mean(window_vals)) if window_vals else float("nan")

    # Neighbor count within 5 Å of target CA.
    target = coords.get(position)
    neighbors_5a = -1
    if target is not None:
        n = 0
        for other_pos, other_coord in coords.items():
            if other_pos == position:
                continue
            if np.linalg.norm(other_coord - target) <= 5.0:
                n += 1
        neighbors_5a = n

    return StructuralFeatures(
        pLDDT_position=plddt[position],
        pLDDT_window_5=window_mean,
        SASA_position=sasa.get(position, float("nan")),
        secondary_structure=ss.get(position, ""),
        neighbors_5A=neighbors_5a,
    )


def compute_all_structural(
    *,
    pdb_dir: Path,
    uniprot_positions: pd.DataFrame,
    progress: bool = True,
) -> pd.DataFrame:
    """Vectorize structural-feature extraction over many variants.

    `uniprot_positions` must have columns `uniprot_id`, `protein_position`
    (and optionally `variant_key` which is passed through).
    Returns a frame with all `_NAN_FEATURES` columns + the pass-through
    `variant_key` if present.
    """
    required = {"uniprot_id", "protein_position"}
    missing = required - set(uniprot_positions.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")

    rows: list[dict[str, object]] = []
    n = len(uniprot_positions)
    for i, record in enumerate(uniprot_positions.itertuples(index=False), 1):
        uni = record.uniprot_id
        pos = int(record.protein_position)
        pdb_path = pdb_dir / f"AF-{uni}-F1-model_v4.pdb"
        feats = compute_for_position(pdb_path, pos)
        out = feats.as_row(uni, pos)
        vk = getattr(record, "variant_key", None)
        if vk is not None:
            out["variant_key"] = vk
        rows.append(out)
        if progress and (i % 500 == 0 or i == n):
            print(f"  [structural] {i:,}/{n:,}")

    return pd.DataFrame(rows)
