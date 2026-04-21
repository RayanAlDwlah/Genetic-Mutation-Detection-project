"""Fetch SIFT and PolyPhen-2 scores from the Ensembl VEP REST API.

Rationale
---------
SIFT and PolyPhen-2 have been the standard missense-effect predictors
for two decades; they're the first thing a reviewer asks about. They
are trivially accessible through the VEP REST response (both scores
arrive in the `transcript_consequences` block for the canonical
missense transcript), so we don't need to download any large table.

Score convention
----------------
- SIFT: **lower = more damaging** (0 = damaging, 1 = tolerated).
  We invert the sign for the evaluation pipeline (which expects
  "higher = pathogenic") via `BaselineMetadata.higher_is_pathogenic = False`.
- PolyPhen-2: **higher = more damaging** (0 = benign, 1 = probably damaging).
  Standard orientation.

Coverage
--------
VEP returns SIFT/PolyPhen scores only for variants where the canonical
transcript has an annotated missense consequence. Synonymous / splice /
UTR variants and genes without SIFT/PolyPhen annotations return NaN.
We record coverage honestly as part of the output.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

# GRCh38 endpoint — our ClinVar-derived splits are on GRCh38 despite the
# `dbNSFP5.3.1a_grch37.gz` source filename. We verified empirically:
# sampling 5 test variants, all 5 matched their annotated gene + amino
# acid change on GRCh38 but not on GRCh37 (the GRCh37 endpoint returned
# unrelated intron / downstream consequences at the same numeric position).
# See `docs/CHANGELOG.md` "Build correction" for the full audit.
#
# External (denovo-db) variants use a *different* REST URL because
# denovo-db itself publishes GRCh37 coordinates — those go through the
# GRCh37 endpoint in `src/esm2_scorer.py`.
VEP_URL = "https://rest.ensembl.org/vep/human/region"
BATCH_SIZE = 200
SLEEP = 0.6
MAX_RETRIES = 4


@dataclass(frozen=True)
class SiftPolyphenLookup:
    scores: pd.DataFrame  # columns: variant_key, sift_score, polyphen_score
    coverage: float


def _variant_token(row: pd.Series) -> str:
    return f"{row['chr']} {row['pos']} . {row['ref']} {row['alt']}"


def _post_vep(variants: list[str]) -> list[dict]:
    payload = {"variants": variants}
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                VEP_URL,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                data=json.dumps(payload),
                timeout=90,
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"VEP POST failed after {MAX_RETRIES} retries")


def _pick_canonical_missense(tcs: list[dict]) -> dict | None:
    canonical = [
        t
        for t in tcs
        if t.get("canonical") == 1 and "missense_variant" in (t.get("consequence_terms") or [])
    ]
    if canonical:
        return canonical[0]
    any_mis = [t for t in tcs if "missense_variant" in (t.get("consequence_terms") or [])]
    return any_mis[0] if any_mis else None


def fetch_sift_polyphen(
    query_df: pd.DataFrame,
    *,
    cache_path: Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> SiftPolyphenLookup:
    """Query VEP REST for SIFT + PolyPhen-2 scores for every row in `query_df`.

    `query_df` must have `chr, pos, ref, alt, variant_key`. Returns a
    lookup DataFrame with NaN entries for unscored variants.

    A resumable parquet cache at `cache_path` lets us interrupt and
    resume long fetches across multiple sessions — the cache is keyed
    by `variant_key`.
    """
    cached: dict[str, dict] = {}
    if cache_path is not None and cache_path.exists() and not force:
        cdf = pd.read_parquet(cache_path)
        cached = {r["variant_key"]: r.to_dict() for _, r in cdf.iterrows()}

    tokens = query_df.apply(_variant_token, axis=1).tolist()
    keys = query_df["variant_key"].tolist()
    need_idx = [i for i, k in enumerate(keys) if k not in cached]

    if progress:
        print(
            f"[SIFT/PolyPhen] {len(cached):,} cached, {len(need_idx):,} to fetch "
            f"(~{len(need_idx) / BATCH_SIZE * (SLEEP + 1):.0f} s estimated)"
        )

    n_batches = (len(need_idx) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi, start in enumerate(range(0, len(need_idx), BATCH_SIZE), 1):
        sub = need_idx[start : start + BATCH_SIZE]
        batch_tokens = [tokens[i] for i in sub]
        batch_keys = [keys[i] for i in sub]
        recs = _post_vep(batch_tokens)
        for vk, rec in zip(batch_keys, recs, strict=False):
            t = _pick_canonical_missense(rec.get("transcript_consequences") or [])
            if t is None:
                cached[vk] = {"variant_key": vk, "sift_score": None, "polyphen_score": None}
                continue
            cached[vk] = {
                "variant_key": vk,
                "sift_score": t.get("sift_score"),
                "polyphen_score": t.get("polyphen_score"),
            }
        if progress and (bi % 5 == 0 or bi == n_batches):
            print(f"  [SIFT/PolyPhen] batch {bi}/{n_batches}")
        time.sleep(SLEEP)

    scores = pd.DataFrame([cached[k] for k in keys if k in cached])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(list(cached.values())).to_parquet(cache_path, index=False)

    n_covered = int(scores[["sift_score", "polyphen_score"]].notna().any(axis=1).sum())
    return SiftPolyphenLookup(scores=scores, coverage=n_covered / max(len(query_df), 1))
