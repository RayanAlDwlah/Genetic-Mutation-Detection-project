#!/usr/bin/env python3
"""Extract selected dbNSFP features with circularity protection.

PURPOSE:
- Extract pre-computed and derived variant features from dbNSFP.
- Keep only variants matching cleaned ClinVar (optional but recommended).
- Enforce circularity protection by excluding ClinVar-derived meta-predictors.

Example:
    python src/dbnsfp_extraction.py \
      --input data/raw/dbnsfp/dbNSFP5.3.1a_grch37.gz \
      --clinvar-variants data/intermediate/clinvar_labeled_clean.parquet
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from src.output import echo
from src.utils import load_yaml_config, normalize_chromosome, resolve_path


DEFAULT_OUTPUT = "data/intermediate/dbnsfp_selected_features.parquet"


# Standard amino-acid physicochemical values used for derived features.
# hydrophobicity: Kyte-Doolittle scale
# molecular_weight: residue molecular weight (Da)
# pI: isoelectric point
# volume: approximate side-chain volume
# polarity: relative polarity scale
# charge: nominal residue charge near neutral pH
AMINO_ACID_PROPERTIES = {
    "A": {"hydrophobicity": 1.8, "molecular_weight": 89.09, "pI": 6.00, "volume": 88.6, "polarity": 8.1, "charge": 0.0},
    "R": {"hydrophobicity": -4.5, "molecular_weight": 174.20, "pI": 10.76, "volume": 173.4, "polarity": 10.5, "charge": 1.0},
    "N": {"hydrophobicity": -3.5, "molecular_weight": 132.12, "pI": 5.41, "volume": 114.1, "polarity": 11.6, "charge": 0.0},
    "D": {"hydrophobicity": -3.5, "molecular_weight": 133.10, "pI": 2.77, "volume": 111.1, "polarity": 13.0, "charge": -1.0},
    "C": {"hydrophobicity": 2.5, "molecular_weight": 121.15, "pI": 5.07, "volume": 108.5, "polarity": 5.5, "charge": 0.0},
    "Q": {"hydrophobicity": -3.5, "molecular_weight": 146.15, "pI": 5.65, "volume": 143.8, "polarity": 10.5, "charge": 0.0},
    "E": {"hydrophobicity": -3.5, "molecular_weight": 147.13, "pI": 3.22, "volume": 138.4, "polarity": 12.3, "charge": -1.0},
    "G": {"hydrophobicity": -0.4, "molecular_weight": 75.07, "pI": 5.97, "volume": 60.1, "polarity": 9.0, "charge": 0.0},
    "H": {"hydrophobicity": -3.2, "molecular_weight": 155.16, "pI": 7.59, "volume": 153.2, "polarity": 10.4, "charge": 0.0},
    "I": {"hydrophobicity": 4.5, "molecular_weight": 131.17, "pI": 6.02, "volume": 166.7, "polarity": 5.2, "charge": 0.0},
    "L": {"hydrophobicity": 3.8, "molecular_weight": 131.17, "pI": 5.98, "volume": 166.7, "polarity": 4.9, "charge": 0.0},
    "K": {"hydrophobicity": -3.9, "molecular_weight": 146.19, "pI": 9.74, "volume": 168.6, "polarity": 11.3, "charge": 1.0},
    "M": {"hydrophobicity": 1.9, "molecular_weight": 149.21, "pI": 5.74, "volume": 162.9, "polarity": 5.7, "charge": 0.0},
    "F": {"hydrophobicity": 2.8, "molecular_weight": 165.19, "pI": 5.48, "volume": 189.9, "polarity": 5.2, "charge": 0.0},
    "P": {"hydrophobicity": -1.6, "molecular_weight": 115.13, "pI": 6.30, "volume": 112.7, "polarity": 8.0, "charge": 0.0},
    "S": {"hydrophobicity": -0.8, "molecular_weight": 105.09, "pI": 5.68, "volume": 89.0, "polarity": 9.2, "charge": 0.0},
    "T": {"hydrophobicity": -0.7, "molecular_weight": 119.12, "pI": 5.60, "volume": 116.1, "polarity": 8.6, "charge": 0.0},
    "W": {"hydrophobicity": -0.9, "molecular_weight": 204.23, "pI": 5.89, "volume": 227.8, "polarity": 5.4, "charge": 0.0},
    "Y": {"hydrophobicity": -1.3, "molecular_weight": 181.19, "pI": 5.66, "volume": 193.6, "polarity": 6.2, "charge": 0.0},
    "V": {"hydrophobicity": 4.2, "molecular_weight": 117.15, "pI": 5.96, "volume": 140.0, "polarity": 5.9, "charge": 0.0},
}

THREE_TO_ONE_AA = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

BLOSUM62_TABLE = {
    "A": {"A": 4, "R": -1, "N": -2, "D": -2, "C": 0, "Q": -1, "E": -1, "G": 0, "H": -2, "I": -1, "L": -1, "K": -1, "M": -1, "F": -2, "P": -1, "S": 1, "T": 0, "W": -3, "Y": -2, "V": 0},
    "R": {"A": -1, "R": 5, "N": 0, "D": -2, "C": -3, "Q": 1, "E": 0, "G": -2, "H": 0, "I": -3, "L": -2, "K": 2, "M": -1, "F": -3, "P": -2, "S": -1, "T": -1, "W": -3, "Y": -2, "V": -3},
    "N": {"A": -2, "R": 0, "N": 6, "D": 1, "C": -3, "Q": 0, "E": 0, "G": 0, "H": 1, "I": -3, "L": -3, "K": 0, "M": -2, "F": -3, "P": -2, "S": 1, "T": 0, "W": -4, "Y": -2, "V": -3},
    "D": {"A": -2, "R": -2, "N": 1, "D": 6, "C": -3, "Q": 0, "E": 2, "G": -1, "H": -1, "I": -3, "L": -4, "K": -1, "M": -3, "F": -3, "P": -1, "S": 0, "T": -1, "W": -4, "Y": -3, "V": -3},
    "C": {"A": 0, "R": -3, "N": -3, "D": -3, "C": 9, "Q": -3, "E": -4, "G": -3, "H": -3, "I": -1, "L": -1, "K": -3, "M": -1, "F": -2, "P": -3, "S": -1, "T": -1, "W": -2, "Y": -2, "V": -1},
    "Q": {"A": -1, "R": 1, "N": 0, "D": 0, "C": -3, "Q": 5, "E": 2, "G": -2, "H": 0, "I": -3, "L": -2, "K": 1, "M": 0, "F": -3, "P": -1, "S": 0, "T": -1, "W": -2, "Y": -1, "V": -2},
    "E": {"A": -1, "R": 0, "N": 0, "D": 2, "C": -4, "Q": 2, "E": 5, "G": -2, "H": 0, "I": -3, "L": -3, "K": 1, "M": -2, "F": -3, "P": -1, "S": 0, "T": -1, "W": -3, "Y": -2, "V": -2},
    "G": {"A": 0, "R": -2, "N": 0, "D": -1, "C": -3, "Q": -2, "E": -2, "G": 6, "H": -2, "I": -4, "L": -4, "K": -2, "M": -3, "F": -3, "P": -2, "S": 0, "T": -2, "W": -2, "Y": -3, "V": -3},
    "H": {"A": -2, "R": 0, "N": 1, "D": -1, "C": -3, "Q": 0, "E": 0, "G": -2, "H": 8, "I": -3, "L": -3, "K": -1, "M": -2, "F": -1, "P": -2, "S": -1, "T": -2, "W": -2, "Y": 2, "V": -3},
    "I": {"A": -1, "R": -3, "N": -3, "D": -3, "C": -1, "Q": -3, "E": -3, "G": -4, "H": -3, "I": 4, "L": 2, "K": -3, "M": 1, "F": 0, "P": -3, "S": -2, "T": -1, "W": -3, "Y": -1, "V": 3},
    "L": {"A": -1, "R": -2, "N": -3, "D": -4, "C": -1, "Q": -2, "E": -3, "G": -4, "H": -3, "I": 2, "L": 4, "K": -2, "M": 2, "F": 0, "P": -3, "S": -2, "T": -1, "W": -2, "Y": -1, "V": 1},
    "K": {"A": -1, "R": 2, "N": 0, "D": -1, "C": -3, "Q": 1, "E": 1, "G": -2, "H": -1, "I": -3, "L": -2, "K": 5, "M": -1, "F": -3, "P": -1, "S": 0, "T": -1, "W": -3, "Y": -2, "V": -2},
    "M": {"A": -1, "R": -1, "N": -2, "D": -3, "C": -1, "Q": 0, "E": -2, "G": -3, "H": -2, "I": 1, "L": 2, "K": -1, "M": 5, "F": 0, "P": -2, "S": -1, "T": -1, "W": -1, "Y": -1, "V": 1},
    "F": {"A": -2, "R": -3, "N": -3, "D": -3, "C": -2, "Q": -3, "E": -3, "G": -3, "H": -1, "I": 0, "L": 0, "K": -3, "M": 0, "F": 6, "P": -4, "S": -2, "T": -2, "W": 1, "Y": 3, "V": -1},
    "P": {"A": -1, "R": -2, "N": -2, "D": -1, "C": -3, "Q": -1, "E": -1, "G": -2, "H": -2, "I": -3, "L": -3, "K": -1, "M": -2, "F": -4, "P": 7, "S": -1, "T": -1, "W": -4, "Y": -3, "V": -2},
    "S": {"A": 1, "R": -1, "N": 1, "D": 0, "C": -1, "Q": 0, "E": 0, "G": 0, "H": -1, "I": -2, "L": -2, "K": 0, "M": -1, "F": -2, "P": -1, "S": 4, "T": 1, "W": -3, "Y": -2, "V": -2},
    "T": {"A": 0, "R": -1, "N": 0, "D": -1, "C": -1, "Q": -1, "E": -1, "G": -2, "H": -2, "I": -1, "L": -1, "K": -1, "M": -1, "F": -2, "P": -1, "S": 1, "T": 5, "W": -2, "Y": -2, "V": 0},
    "W": {"A": -3, "R": -3, "N": -4, "D": -4, "C": -2, "Q": -2, "E": -3, "G": -2, "H": -2, "I": -3, "L": -2, "K": -3, "M": -1, "F": 1, "P": -4, "S": -3, "T": -2, "W": 11, "Y": 2, "V": -3},
    "Y": {"A": -2, "R": -2, "N": -2, "D": -3, "C": -2, "Q": -1, "E": -2, "G": -3, "H": 2, "I": -1, "L": -1, "K": -2, "M": -1, "F": 3, "P": -3, "S": -2, "T": -2, "W": 2, "Y": 7, "V": -1},
    "V": {"A": 0, "R": -3, "N": -3, "D": -3, "C": -1, "Q": -2, "E": -2, "G": -3, "H": -3, "I": 3, "L": 1, "K": -2, "M": 1, "F": -1, "P": -2, "S": -2, "T": 0, "W": -3, "Y": -1, "V": 4},
}

GRANTHAM_TABLE = {
    "A": {"A": 0, "R": 112, "N": 111, "D": 126, "C": 195, "Q": 91, "E": 107, "G": 60, "H": 86, "I": 94, "L": 96, "K": 106, "M": 84, "F": 113, "P": 27, "S": 99, "T": 58, "W": 148, "Y": 112, "V": 64},
    "R": {"A": 112, "R": 0, "N": 86, "D": 96, "C": 180, "Q": 43, "E": 54, "G": 125, "H": 29, "I": 97, "L": 102, "K": 26, "M": 91, "F": 97, "P": 103, "S": 110, "T": 71, "W": 101, "Y": 77, "V": 96},
    "N": {"A": 111, "R": 86, "N": 0, "D": 23, "C": 139, "Q": 46, "E": 42, "G": 80, "H": 68, "I": 149, "L": 153, "K": 94, "M": 142, "F": 158, "P": 91, "S": 46, "T": 65, "W": 174, "Y": 143, "V": 133},
    "D": {"A": 126, "R": 96, "N": 23, "D": 0, "C": 154, "Q": 61, "E": 45, "G": 94, "H": 81, "I": 168, "L": 172, "K": 101, "M": 160, "F": 177, "P": 108, "S": 65, "T": 85, "W": 181, "Y": 160, "V": 152},
    "C": {"A": 195, "R": 180, "N": 139, "D": 154, "C": 0, "Q": 154, "E": 170, "G": 159, "H": 174, "I": 198, "L": 198, "K": 202, "M": 196, "F": 205, "P": 169, "S": 112, "T": 149, "W": 215, "Y": 194, "V": 192},
    "Q": {"A": 91, "R": 43, "N": 46, "D": 61, "C": 154, "Q": 0, "E": 29, "G": 87, "H": 24, "I": 109, "L": 113, "K": 53, "M": 101, "F": 116, "P": 76, "S": 68, "T": 42, "W": 130, "Y": 99, "V": 96},
    "E": {"A": 107, "R": 54, "N": 42, "D": 45, "C": 170, "Q": 29, "E": 0, "G": 98, "H": 40, "I": 134, "L": 138, "K": 56, "M": 126, "F": 140, "P": 93, "S": 80, "T": 65, "W": 152, "Y": 122, "V": 121},
    "G": {"A": 60, "R": 125, "N": 80, "D": 94, "C": 159, "Q": 87, "E": 98, "G": 0, "H": 98, "I": 135, "L": 138, "K": 127, "M": 127, "F": 153, "P": 42, "S": 56, "T": 59, "W": 184, "Y": 147, "V": 109},
    "H": {"A": 86, "R": 29, "N": 68, "D": 81, "C": 174, "Q": 24, "E": 40, "G": 98, "H": 0, "I": 94, "L": 99, "K": 32, "M": 87, "F": 100, "P": 77, "S": 89, "T": 47, "W": 115, "Y": 83, "V": 84},
    "I": {"A": 94, "R": 97, "N": 149, "D": 168, "C": 198, "Q": 109, "E": 134, "G": 135, "H": 94, "I": 0, "L": 5, "K": 102, "M": 10, "F": 21, "P": 95, "S": 142, "T": 89, "W": 61, "Y": 33, "V": 29},
    "L": {"A": 96, "R": 102, "N": 153, "D": 172, "C": 198, "Q": 113, "E": 138, "G": 138, "H": 99, "I": 5, "L": 0, "K": 107, "M": 15, "F": 22, "P": 98, "S": 145, "T": 92, "W": 61, "Y": 36, "V": 32},
    "K": {"A": 106, "R": 26, "N": 94, "D": 101, "C": 202, "Q": 53, "E": 56, "G": 127, "H": 32, "I": 102, "L": 107, "K": 0, "M": 95, "F": 102, "P": 103, "S": 121, "T": 78, "W": 110, "Y": 85, "V": 97},
    "M": {"A": 84, "R": 91, "N": 142, "D": 160, "C": 196, "Q": 101, "E": 126, "G": 127, "H": 87, "I": 10, "L": 15, "K": 95, "M": 0, "F": 28, "P": 87, "S": 135, "T": 81, "W": 67, "Y": 36, "V": 21},
    "F": {"A": 113, "R": 97, "N": 158, "D": 177, "C": 205, "Q": 116, "E": 140, "G": 153, "H": 100, "I": 21, "L": 22, "K": 102, "M": 28, "F": 0, "P": 114, "S": 155, "T": 103, "W": 40, "Y": 22, "V": 50},
    "P": {"A": 27, "R": 103, "N": 91, "D": 108, "C": 169, "Q": 76, "E": 93, "G": 42, "H": 77, "I": 95, "L": 98, "K": 103, "M": 87, "F": 114, "P": 0, "S": 74, "T": 38, "W": 147, "Y": 110, "V": 68},
    "S": {"A": 99, "R": 110, "N": 46, "D": 65, "C": 112, "Q": 68, "E": 80, "G": 56, "H": 89, "I": 142, "L": 145, "K": 121, "M": 135, "F": 155, "P": 74, "S": 0, "T": 58, "W": 177, "Y": 144, "V": 124},
    "T": {"A": 58, "R": 71, "N": 65, "D": 85, "C": 149, "Q": 42, "E": 65, "G": 59, "H": 47, "I": 89, "L": 92, "K": 78, "M": 81, "F": 103, "P": 38, "S": 58, "T": 0, "W": 128, "Y": 92, "V": 69},
    "W": {"A": 148, "R": 101, "N": 174, "D": 181, "C": 215, "Q": 130, "E": 152, "G": 184, "H": 115, "I": 61, "L": 61, "K": 110, "M": 67, "F": 40, "P": 147, "S": 177, "T": 128, "W": 0, "Y": 37, "V": 88},
    "Y": {"A": 112, "R": 77, "N": 143, "D": 160, "C": 194, "Q": 99, "E": 122, "G": 147, "H": 83, "I": 33, "L": 36, "K": 85, "M": 36, "F": 22, "P": 110, "S": 144, "T": 92, "W": 37, "Y": 0, "V": 55},
    "V": {"A": 64, "R": 96, "N": 133, "D": 152, "C": 192, "Q": 96, "E": 121, "G": 109, "H": 84, "I": 29, "L": 32, "K": 97, "M": 21, "F": 50, "P": 68, "S": 124, "T": 69, "W": 88, "Y": 55, "V": 0},
}

def normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def first_token(value: Any) -> Any:
    """dbNSFP has '.' and multi-valued fields; keep first representative token."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan

    text = str(value).strip()
    if not text or text in {".", "NA", "N/A", "nan", "None"}:
        return np.nan

    token = text.split(";", 1)[0].strip()
    token = token.split(",", 1)[0].strip()
    if not token or token in {".", "NA", "N/A", "nan", "None"}:
        return np.nan
    return token

def normalize_aa(value: Any) -> str | None:
    token = first_token(value)
    if token is np.nan or pd.isna(token):
        return None

    text = str(token).strip().upper().replace("*", "")
    if not text:
        return None

    if len(text) == 1 and text in AMINO_ACID_PROPERTIES:
        return text

    if len(text) == 3 and text in THREE_TO_ONE_AA:
        return THREE_TO_ONE_AA[text]

    return None


def matrix_lookup(matrix: dict[str, dict[str, float]], ref_aa: Any, alt_aa: Any) -> float | None:
    ref = normalize_aa(ref_aa)
    alt = normalize_aa(alt_aa)
    if ref is None or alt is None:
        return None
    return matrix.get(ref, {}).get(alt)


def first_non_null(series: pd.Series) -> Any:
    for value in series:
        if pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return np.nan

def infer_input_format(input_path: Path, input_format: str) -> str:
    if input_format != "auto":
        return input_format

    name = input_path.name.lower()
    if name.endswith((".parquet", ".tsv", ".tsv.gz", ".csv", ".csv.gz", ".txt", ".txt.gz")):
        return "tsv"
    if name.endswith((".gz", ".bgz", ".tab.gz")):
        return "tabix"
    return "tsv"


def get_text_header_columns(path: Path, delimiter: str) -> list[str]:
    import gzip

    opener = gzip.open if path.suffix in {".gz", ".bgz"} or str(path).endswith(".tab.gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline().rstrip("\n")

    if not first_line:
        raise ValueError(f"Input file appears empty: {path}")

    return first_line.split(delimiter)


def load_clinvar_variant_set(path: Path) -> set[str]:
    df = pd.read_parquet(path, columns=["variant_key"])
    variants = set(df["variant_key"].dropna().astype(str))
    echo(f"Loaded ClinVar variants: {len(variants):,}")
    return variants


def resolve_columns(available: list[str], candidates_map: dict[str, Any]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    norm_to_actual = {normalize_name(col): col for col in available}

    for canonical, candidates in candidates_map.items():
        chosen = None
        for candidate in ensure_list(candidates):
            if candidate in available:
                chosen = candidate
                break
            norm = normalize_name(candidate)
            if norm in norm_to_actual:
                chosen = norm_to_actual[norm]
                break
        out[canonical] = chosen

    return out


def preprocess_series(series: pd.Series) -> pd.Series:
    return series.map(first_token)


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(preprocess_series(series), errors="coerce").astype(float)


def pick_separator(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return ","
    return "\t"


def extract_from_chunks(
    input_path: Path,
    input_format: str,
    clinvar_variants: set[str] | None,
    key_cols: dict[str, str | None],
    aa_cols: dict[str, str | None],
    feature_cols: dict[str, str | None],
    chunk_size: int = 200_000,
) -> tuple[pd.DataFrame, list[str]]:
    """Read dbNSFP in chunks and return extracted rows + source columns used."""
    used_source_columns = sorted({c for c in ([*key_cols.values(), *aa_cols.values(), *feature_cols.values()]) if c})

    if input_path.suffix == ".parquet":
        chunk_iter = [pd.read_parquet(input_path, columns=used_source_columns)]
    else:
        sep = pick_separator(input_path)
        chunk_iter = pd.read_csv(
            input_path,
            sep=sep,
            dtype=str,
            usecols=lambda col: col in set(used_source_columns),
            chunksize=chunk_size,
            low_memory=False,
        )

    extracted_frames: list[pd.DataFrame] = []
    processed_rows = 0

    for chunk_idx, chunk in enumerate(chunk_iter, start=1):
        processed_rows += len(chunk)

        chr_col = key_cols.get("chr")
        pos_col = key_cols.get("pos")
        ref_col = key_cols.get("ref")
        alt_col = key_cols.get("alt")

        if not all([chr_col, pos_col, ref_col, alt_col]):
            raise ValueError("Missing required key columns in dbNSFP input")

        out = pd.DataFrame(index=chunk.index)
        out["chr"] = chunk[chr_col].map(normalize_chromosome)
        out["pos"] = pd.to_numeric(preprocess_series(chunk[pos_col]), errors="coerce")
        out["ref"] = preprocess_series(chunk[ref_col]).astype("string").str.upper().replace({"<NA>": np.nan})
        out["alt"] = preprocess_series(chunk[alt_col]).astype("string").str.upper().replace({"<NA>": np.nan})

        out = out[
            out["chr"].notna()
            & out["pos"].notna()
            & out["ref"].notna()
            & out["alt"].notna()
        ].copy()

        out["pos"] = out["pos"].astype(int)
        out["variant_key"] = (
            out["chr"].astype(str)
            + ":"
            + out["pos"].astype(str)
            + ":"
            + out["ref"].astype(str)
            + ":"
            + out["alt"].astype(str)
        )

        if clinvar_variants is not None:
            out = out[out["variant_key"].isin(clinvar_variants)].copy()
            if out.empty:
                if chunk_idx % 10 == 0:
                    echo(f"Chunk {chunk_idx:,}: processed={processed_rows:,}, matched=0")
                continue

        ref_aa_col = aa_cols.get("ref_aa")
        alt_aa_col = aa_cols.get("alt_aa")
        out["ref_aa"] = chunk.loc[out.index, ref_aa_col].map(normalize_aa) if ref_aa_col else None
        out["alt_aa"] = chunk.loc[out.index, alt_aa_col].map(normalize_aa) if alt_aa_col else None

        for feature_name, source_col in feature_cols.items():
            if source_col is None:
                out[feature_name] = np.nan
                continue

            source_series = chunk.loc[out.index, source_col]

            if feature_name in {
                "secondary_structure_prediction",
                "solvent_accessibility",
            }:
                cleaned = preprocess_series(source_series)
                if feature_name == "secondary_structure_prediction":
                    cleaned = cleaned.astype("string").str.upper().str.extract(r"([HEC])", expand=False)
                out[feature_name] = cleaned
            elif feature_name == "pfam_domain":
                cleaned = preprocess_series(source_series)
                out[feature_name] = cleaned.notna() & cleaned.astype(str).str.strip().ne("")
            else:
                out[feature_name] = as_numeric(source_series)

        # Derived amino-acid properties (always computed from ref_aa / alt_aa).
        for suffix, prop in [("hydrophobicity", "hydrophobicity"), ("molecular_weight", "molecular_weight"), ("pI", "pI"), ("volume", "volume"), ("polarity", "polarity"), ("charge", "charge")]:
            out[f"{suffix}_ref"] = out["ref_aa"].map(lambda aa: AMINO_ACID_PROPERTIES.get(aa, {}).get(prop) if aa else np.nan)
            out[f"{suffix}_alt"] = out["alt_aa"].map(lambda aa: AMINO_ACID_PROPERTIES.get(aa, {}).get(prop) if aa else np.nan)

        out["polarity_change"] = out["polarity_alt"] - out["polarity_ref"]
        out["volume_change"] = out["volume_alt"] - out["volume_ref"]
        out["charge_change"] = out["charge_alt"] - out["charge_ref"]

        # Compute BLOSUM62/Grantham if source columns are missing or NaN.
        if "BLOSUM62_score" in out.columns:
            computed = out.apply(lambda r: matrix_lookup(BLOSUM62_TABLE, r.get("ref_aa"), r.get("alt_aa")), axis=1)
            out["BLOSUM62_score"] = out["BLOSUM62_score"].where(out["BLOSUM62_score"].notna(), computed)

        if "Grantham_distance" in out.columns:
            computed = out.apply(lambda r: matrix_lookup(GRANTHAM_TABLE, r.get("ref_aa"), r.get("alt_aa")), axis=1)
            out["Grantham_distance"] = out["Grantham_distance"].where(out["Grantham_distance"].notna(), computed)

        extracted_frames.append(out)

        if chunk_idx % 5 == 0:
            echo(
                f"Chunk {chunk_idx:,}: processed={processed_rows:,}, "
                f"matched={sum(len(df) for df in extracted_frames):,}"
            )

    if not extracted_frames:
        return pd.DataFrame(columns=["variant_key", "ref_aa", "alt_aa"]), used_source_columns

    full_df = pd.concat(extracted_frames, ignore_index=True)
    return full_df, used_source_columns


def aggregate_by_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse transcript-level duplicates to one row per variant_key."""
    if df.empty:
        return df

    agg_map: dict[str, Any] = {}
    for col in df.columns:
        if col == "variant_key":
            continue
        if col == "pfam_domain":
            agg_map[col] = lambda s: bool(pd.Series(s).fillna(False).astype(bool).any())
        else:
            agg_map[col] = first_non_null

    out = df.groupby("variant_key", as_index=False).agg(agg_map)
    return out


def apply_missingness_policy(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    """Log >50% missing and drop >80% missing feature columns."""
    if df.empty:
        return df, {}, []

    keep_base = {"variant_key", "ref_aa", "alt_aa"}
    feature_cols = [c for c in df.columns if c not in keep_base]

    missing_summary: dict[str, float] = {}
    high_missing_50: list[str] = []
    drop_cols_80: list[str] = []

    for col in feature_cols:
        missing_pct = float(df[col].isna().mean() * 100.0)
        missing_summary[col] = round(missing_pct, 4)
        if missing_pct > 50.0:
            high_missing_50.append(col)
        if missing_pct > 80.0:
            drop_cols_80.append(col)

    if high_missing_50:
        echo("Columns with >50% missing values:")
        for col in high_missing_50:
            echo(f"  - {col}: {missing_summary[col]:.2f}%")

    if drop_cols_80:
        echo("Dropping columns with >80% missing values:")
        for col in drop_cols_80:
            echo(f"  - {col}: {missing_summary[col]:.2f}%")
        df = df.drop(columns=drop_cols_80)

    return df, missing_summary, drop_cols_80


def check_circularity(columns: list[str], excluded_predictors: list[str]) -> list[str]:
    """Detect accidental inclusion of circular ClinVar-derived predictors."""
    detected: list[str] = []
    for col in columns:
        normalized_col = normalize_name(col)
        for predictor in excluded_predictors:
            if normalize_name(predictor) and normalize_name(predictor) in normalized_col:
                detected.append(col)
                break
    return sorted(set(detected))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract selected dbNSFP features")
    parser.add_argument("--input", required=True, help="Path to dbNSFP file")
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "tabix", "tsv"],
        help="Input format (default: auto-detect)",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output parquet path")
    parser.add_argument("--clinvar-variants", default=None, help="Path to clinvar_labeled_clean.parquet")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    input_path = resolve_path(repo_root, args.input)
    output_path = resolve_path(repo_root, args.output)
    config_path = resolve_path(repo_root, args.config)

    if not input_path.exists():
        raise FileNotFoundError(f"dbNSFP input not found: {input_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_yaml_config(config_path)
    feature_cfg = cfg.get("dbnsfp_features")
    if not isinstance(feature_cfg, dict):
        raise ValueError("Missing 'dbnsfp_features' section in config.yaml")

    input_format = infer_input_format(input_path, args.input_format)

    key_cfg = feature_cfg.get("key_columns", {}) or {}
    conservation_cfg = feature_cfg.get("conservation", {}) or {}
    physicochemical_cfg = feature_cfg.get("physicochemical", {}) or {}
    structural_cfg = feature_cfg.get("structural", {}) or {}

    feature_candidates: dict[str, Any] = {}
    feature_candidates.update(conservation_cfg)
    feature_candidates.update(physicochemical_cfg)
    feature_candidates.update(structural_cfg)

    excluded_predictors = [str(x) for x in feature_cfg.get("exclude_predictors", [])]

    available_columns = get_text_header_columns(input_path, delimiter=pick_separator(input_path))
    key_cols = resolve_columns(available_columns, key_cfg)

    aa_candidate_cfg = {
        "ref_aa": key_cfg.get("ref_aa", ["aaref"]),
        "alt_aa": key_cfg.get("alt_aa", ["aaalt"]),
    }
    aa_cols = resolve_columns(available_columns, aa_candidate_cfg)
    feature_cols = resolve_columns(available_columns, feature_candidates)

    echo(f"Input: {input_path}")
    echo(f"Input format: {input_format}")
    echo(f"Output: {output_path}")
    echo(f"Config: {config_path}")

    clinvar_variants = None
    if args.clinvar_variants:
        clinvar_path = resolve_path(repo_root, args.clinvar_variants)
        if not clinvar_path.exists():
            raise FileNotFoundError(f"ClinVar variants file not found: {clinvar_path}")
        clinvar_variants = load_clinvar_variant_set(clinvar_path)

    extracted_df, used_source_columns = extract_from_chunks(
        input_path=input_path,
        input_format=input_format,
        clinvar_variants=clinvar_variants,
        key_cols=key_cols,
        aa_cols=aa_cols,
        feature_cols=feature_cols,
    )

    extracted_df = aggregate_by_variant(extracted_df)

    # Keep required output layout.
    base_cols = ["variant_key", "ref_aa", "alt_aa"]
    all_feature_cols = [c for c in extracted_df.columns if c not in base_cols]

    detected_circular = check_circularity(all_feature_cols + used_source_columns, excluded_predictors)
    if detected_circular:
        echo("WARNING: Detected excluded circularity predictors in extracted columns:")
        for col in detected_circular:
            echo(f"  - {col}")
        extracted_df = extracted_df.drop(columns=[c for c in detected_circular if c in extracted_df.columns], errors="ignore")

    extracted_df, missing_summary, dropped_high_missing = apply_missingness_policy(extracted_df)

    # Ensure numeric dtype for quantitative columns.
    final_feature_cols = [c for c in extracted_df.columns if c not in base_cols]
    for col in final_feature_cols:
        if col in {"secondary_structure_prediction", "solvent_accessibility", "pfam_domain"}:
            continue
        extracted_df[col] = pd.to_numeric(extracted_df[col], errors="coerce")

    if "pfam_domain" in extracted_df.columns:
        extracted_df["pfam_domain"] = extracted_df["pfam_domain"].fillna(False).astype(bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_df.to_parquet(output_path, index=False)

    echo("Extraction finished.")
    echo(f"Total variants extracted: {len(extracted_df):,}")
    echo(f"Features extracted: {len(final_feature_cols)}")
    echo(f"Saved parquet: {output_path}")


if __name__ == "__main__":
    main()
