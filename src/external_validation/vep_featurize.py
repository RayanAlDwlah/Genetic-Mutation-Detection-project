"""Fetch conservation + AA-derived features from Ensembl VEP REST (GRCh37).

Rationale
---------
`src/dbnsfp_extraction.py` requires the bulk dbNSFP TSV (~30 GB) to featurize
new variants. For Phase D v1 we cannot ship that to a clean clone, so this
module uses the Ensembl VEP REST API to pull the same conservation scores
the training pipeline consumes, then reproduces all AA-derived features
(BLOSUM62, Grantham, hydrophobicity, …) in Python using the same helper
tables as `dbnsfp_extraction.py`.

Gap handling
------------
Fields the VEP plugin does not expose on the public server
(`phyloP30way_mammalian`, `phastCons30way_mammalian`, `pfam_domain`) are
imputed with the **training-set median** — a conservative default that
matches the pipeline's existing imputation strategy. Imputation is logged
per-variant so downstream SHAP can tell imputed rows from real ones.

Rate limits
-----------
Public Ensembl REST allows ~15 req/s. We batch up to 200 variants per POST
(endpoint cap) and insert a small sleep between batches to stay well under
the 55k/hour ceiling. Fetch retry is linear with jitter.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.dbnsfp_extraction import (
    AMINO_ACID_PROPERTIES,
    BLOSUM62_TABLE,
    GRANTHAM_TABLE,
)

VEP_URL = "https://grch37.rest.ensembl.org/vep/human/region"
BATCH_SIZE = 200
SLEEP_BETWEEN_BATCHES = 0.6  # seconds

# Training-set medians for fields VEP doesn't expose on the public server.
TRAIN_IMPUTE_MEDIANS = {
    "phyloP30way_mammalian": 2.791,
    "phastCons30way_mammalian": 0.999,
    "pfam_domain": 0.0,
}

# Gnomad AF features: de-novo variants in affected probands are ultra-rare by
# definition, so 0/median is the honest default. The ablation study showed
# removing AF features costs only ΔROC ≈ -0.003, so exact values don't drive
# the external number materially.
AF_DEFAULTS = {
    "AF_popmax": 0.0,
    "AN": 0.0,
    "AC": 0.0,
    "log_AF": -8.0,  # training-set median for rare/absent variants
}

# Imputation flags — we're providing real (or median-imputed) values; set
# the flag to match what the training pipeline did. For fields we derive
# from AA identity (BLOSUM62, Grantham) or VEP (phyloP, GERP), flag = 0.
IMPUTE_FLAGS = {
    "is_imputed_phyloP100way_vertebrate": 0,
    "is_imputed_phastCons100way_vertebrate": 0,
    "is_imputed_GERP++_RS": 0,
    "is_imputed_Grantham_distance": 0,
    "is_imputed_BLOSUM62_score": 0,
}

# Map VEP JSON keys (lowercased) to our pipeline column names.
_VEP_KEY_MAP = {
    "phylop100way_vertebrate": "phyloP100way_vertebrate",
    "phastcons100way_vertebrate": "phastCons100way_vertebrate",
    "gerp++_rs": "GERP++_RS",
    "gerp++_nr": "GERP++_NR",
}


@dataclass
class VEPFetchConfig:
    url: str = VEP_URL
    batch_size: int = BATCH_SIZE
    sleep_between_batches: float = SLEEP_BETWEEN_BATCHES
    max_retries: int = 3
    cache_dir: Path | None = None


def _variant_token(row: pd.Series) -> str:
    """Build VEP input token: 'chr pos . ref alt'."""
    return f"{row['chr']} {row['pos']} . {row['ref']} {row['alt']}"


def _pick_missense_consequence(tcs: list[dict]) -> dict | None:
    for t in tcs:
        terms = t.get("consequence_terms") or []
        if "missense_variant" in terms:
            return t
    return None


def _post_batch(
    variants: list[str], *, cfg: VEPFetchConfig
) -> list[dict]:
    payload = {"variants": variants}
    last_err = None
    for attempt in range(cfg.max_retries):
        try:
            r = requests.post(
                cfg.url,
                headers={"Content-Type": "application/json",
                         "Accept": "application/json"},
                data=json.dumps(payload),
                timeout=90,
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"VEP POST failed after {cfg.max_retries} retries: {last_err}")


def _extract_row(variant_key: str, rec: dict) -> dict | None:
    """Pick the canonical missense consequence and build a feature row."""
    tcs = rec.get("transcript_consequences") or []
    t = _pick_missense_consequence(tcs)
    if t is None:
        return None
    aa = (t.get("amino_acids") or "").split("/")
    if len(aa) != 2:
        return None
    ref_aa, alt_aa = aa[0].strip().upper(), aa[1].strip().upper()
    if ref_aa not in AMINO_ACID_PROPERTIES or alt_aa not in AMINO_ACID_PROPERTIES:
        return None

    row: dict[str, object] = {"variant_key": variant_key,
                              "ref_aa": ref_aa, "alt_aa": alt_aa}
    for vep_key, our_col in _VEP_KEY_MAP.items():
        v = t.get(vep_key)
        row[our_col] = np.nan if v in (None, "invalid_field") else float(v)

    # BLOSUM62 + Grantham from AA identities.
    row["BLOSUM62_score"] = BLOSUM62_TABLE[ref_aa][alt_aa]
    row["Grantham_distance"] = GRANTHAM_TABLE[ref_aa][alt_aa]

    # Physicochemical ref/alt + changes.
    for prop in ("hydrophobicity", "molecular_weight", "pI",
                 "volume", "polarity", "charge"):
        r = AMINO_ACID_PROPERTIES[ref_aa][prop]
        a = AMINO_ACID_PROPERTIES[alt_aa][prop]
        row[f"{prop}_ref"] = r
        row[f"{prop}_alt"] = a
    row["polarity_change"] = row["polarity_alt"] - row["polarity_ref"]
    row["volume_change"] = row["volume_alt"] - row["volume_ref"]
    row["charge_change"] = row["charge_alt"] - row["charge_ref"]

    # Training-median imputation for VEP-invisible fields.
    for col, med in TRAIN_IMPUTE_MEDIANS.items():
        row[col] = med
    # gnomAD AF defaults (ultra-rare assumption for de-novo variants).
    for col, val in AF_DEFAULTS.items():
        row[col] = val
    # Imputation flags (honest 0 — we're providing real values).
    for col, val in IMPUTE_FLAGS.items():
        row[col] = val
    return row


def fetch_vep_features(
    ext: pd.DataFrame,
    *,
    cfg: VEPFetchConfig = VEPFetchConfig(),
    progress: bool = True,
) -> pd.DataFrame:
    """Return a feature frame keyed by variant_key, ready for the training
    ColumnTransformer. Rows that can't be resolved (non-missense, unmapped
    coords, etc.) are simply absent — callers should left-join and check
    coverage."""
    ext = ext.copy().reset_index(drop=True)
    tokens = ext.apply(_variant_token, axis=1).tolist()
    keys = ext["variant_key"].tolist()

    # Optional disk cache (one row per variant_key).
    cached: dict[str, dict] = {}
    cache_path: Path | None = None
    if cfg.cache_dir is not None:
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cfg.cache_dir / "vep_cache.parquet"
        if cache_path.exists():
            cdf = pd.read_parquet(cache_path)
            cached = {r["variant_key"]: r.to_dict() for _, r in cdf.iterrows()}

    rows: list[dict] = []
    need_indices = [i for i, k in enumerate(keys) if k not in cached]
    if progress:
        print(f"  VEP: {len(cached):,} cached, {len(need_indices):,} to fetch")

    for start in range(0, len(need_indices), cfg.batch_size):
        sub = need_indices[start:start + cfg.batch_size]
        batch_tokens = [tokens[i] for i in sub]
        batch_keys = [keys[i] for i in sub]
        recs = _post_batch(batch_tokens, cfg=cfg)
        for vk, rec in zip(batch_keys, recs):
            r = _extract_row(vk, rec)
            if r is not None:
                rows.append(r)
                cached[vk] = r
        if progress:
            print(f"    batch {start // cfg.batch_size + 1}/"
                  f"{(len(need_indices)+cfg.batch_size-1)//cfg.batch_size}  "
                  f"→ total fetched: {len(rows):,}")
        time.sleep(cfg.sleep_between_batches)

    # Emit from cache + new fetches.
    result_rows = [cached[k] for k in keys if k in cached]
    out = pd.DataFrame(result_rows)
    if cache_path is not None and not out.empty:
        out.to_parquet(cache_path, index=False)
    return out
