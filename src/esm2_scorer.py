#!/usr/bin/env python3
"""Zero-shot ESM-2 masked-LM scoring for missense variants.

Concept
-------
For each missense variant at protein position `i` with reference amino acid
`ref` and alternate `alt`, run the ESM-2 encoder on the full protein
sequence with position `i` replaced by the `<mask>` token. The softmax at
that position gives `P(aa_j | context)` for every canonical amino acid.
We record:

    esm2_prob_ref = P(ref | context)
    esm2_prob_alt = P(alt | context)
    esm2_llr      = log P(alt) - log P(ref)

The LLR is the standard zero-shot pathogenicity score used in Brandes
(2023) and the original ESM-1b / ESM-2 papers. Intuition: a variant that
the protein language model considers plausible (high P(alt)) in a
position where it clearly expects the reference (high P(ref)) tends to be
functionally disruptive.

Important: this signal is **orthogonal** to the variant-level features
the tabular baseline already uses (conservation, AA chemistry). It
encodes evolutionary-sequence context — what residues *co-occur* at this
position in related proteins — which the tabular model has no access to.

Pipeline
--------
1. Query Ensembl VEP REST per variant → `(transcript_id, protein_start,
   ref_aa, alt_aa)` for the canonical missense consequence.
2. Fetch canonical protein sequence per unique transcript from Ensembl
   `/sequence/id/?type=protein`.
3. Load `facebook/esm2_t12_35M_UR50D` once, move to MPS if available.
4. For each variant, build masked input around position `protein_start`,
   run a single forward pass, read off the 20-AA softmax.
5. Write parquet with `variant_key, esm2_prob_ref, esm2_prob_alt,
   esm2_llr, transcript_id, protein_position`.

Caching
-------
- VEP per-variant annotations cached under `data/intermediate/esm2/vep_ann.parquet`.
- Protein sequences cached under `data/intermediate/esm2/sequences.parquet`.
- Per-variant ESM-2 scores cached under `data/intermediate/esm2/scores.parquet`.
All three are resumable — rerunning the script skips variants whose ESM
score is already in the cache.

Runtime note
------------
This module imports `torch` and `transformers` which live in the
`~/.venvs/esm2` venv (Python 3.13 with lzma support). The project's main
pyenv 3.11.7 lacks `_lzma` so HuggingFace refuses to load the tokenizer.
Run this module with:

    /Users/ry7vv/.venvs/esm2/bin/python -m src.esm2_scorer …
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "data/intermediate/esm2"

# Default to GRCh38 (our ClinVar-derived splits are GRCh38). For
# external datasets that publish GRCh37 coordinates (e.g. denovo-db),
# pass the GRCh37 URLs explicitly via the `vep_url` / `seq_url` kwargs.
VEP_URL_DEFAULT = "https://rest.ensembl.org/vep/human/region"
SEQ_URL_DEFAULT = "https://rest.ensembl.org/sequence/id/{tid}?type=protein"
VEP_URL_GRCH37 = "https://grch37.rest.ensembl.org/vep/human/region"
SEQ_URL_GRCH37 = "https://grch37.rest.ensembl.org/sequence/id/{tid}?type=protein"
# Module-level aliases kept for backwards compatibility with older callers.
# New code should pass `vep_url=` / `seq_url=` explicitly.
VEP_URL = VEP_URL_DEFAULT
SEQ_URL = SEQ_URL_DEFAULT
ESM_MODEL = "facebook/esm2_t12_35M_UR50D"
BATCH_VEP = 200
SLEEP = 0.6
MAX_RETRIES = 8  # VEP REST is flaky under load; retry generously with exponential backoff.
MAX_BACKOFF_SEC = 60.0  # Cap per-attempt sleep to avoid stalling forever.

# Canonical 20-AA alphabet; ESM tokenizer recognizes each as a single-character token.
AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def _ensure_dir(p: Path) -> None:
    """``mkdir -p`` that tolerates ``p`` being a symlink to an existing dir.

    In Colab we symlink ``data/intermediate/esm2`` → a Drive folder so the
    cache survives session restarts. Python 3.12's
    ``Path.mkdir(exist_ok=True)`` can still raise ``FileExistsError`` on
    such a symlink (the FUSE-backed Drive mount confuses the resolved-
    target check), so we short-circuit when the target is already a
    directory.
    """
    if p.is_dir():
        return
    p.mkdir(parents=True, exist_ok=True)


# ─────────────────────────── VEP annotation ────────────────────────────


def _variant_token(row: pd.Series) -> str:
    return f"{row['chr']} {row['pos']} . {row['ref']} {row['alt']}"


def _post_vep(variants: list[str], *, vep_url: str = VEP_URL) -> list[dict]:
    """POST a batch to Ensembl VEP REST with exponential-backoff retries.

    Returns the parsed JSON list on success. On final failure (after
    ``MAX_RETRIES`` attempts) returns an empty list instead of raising so
    that the caller can skip the batch and continue — the variants stay
    outside the cache and will be retried on the next run. This is
    essential because Ensembl VEP returns transient 502/503/504 errors
    during peak hours; one bad batch should not abort a multi-hour job.
    """
    payload = {"variants": variants}
    last_err: str | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                vep_url,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                data=json.dumps(payload),
                timeout=90,
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                # Transient server-side error — back off and retry.
                last_err = f"HTTP {r.status_code}"
                wait = min(MAX_BACKOFF_SEC, (2**attempt) + random.uniform(0, 1.5))
                time.sleep(wait)
                continue
            # Anything else (400/401/404/…) is a client-side error; retrying won't help.
            last_err = f"HTTP {r.status_code}"
            break
        except requests.RequestException as exc:
            last_err = type(exc).__name__
            wait = min(MAX_BACKOFF_SEC, 1.5 * (2**attempt) + random.uniform(0, 1.5))
            time.sleep(wait)
    print(
        f"    ⚠ VEP batch failed after {MAX_RETRIES} attempts "
        f"({last_err}) — skipping {len(variants)} variants this run; "
        f"they remain in the cache-miss queue for the next retry."
    )
    return []


def _pick_canonical_missense(tcs: list[dict]) -> dict | None:
    # Prefer canonical transcript with a missense_variant consequence.
    canonical = [
        t
        for t in tcs
        if t.get("canonical") == 1 and "missense_variant" in (t.get("consequence_terms") or [])
    ]
    if canonical:
        return canonical[0]
    any_mis = [t for t in tcs if "missense_variant" in (t.get("consequence_terms") or [])]
    return any_mis[0] if any_mis else None


def annotate_with_vep(
    ext: pd.DataFrame,
    *,
    cache_path: Path,
    progress: bool = True,
    vep_url: str = VEP_URL_DEFAULT,
    seq_url: str = SEQ_URL_DEFAULT,
) -> pd.DataFrame:
    """Add `transcript_id, protein_position, ref_aa, alt_aa` via VEP.

    Resumable: rows present in the cache parquet are reused verbatim.
    Missing rows are fetched in batches of 200 and appended.

    `vep_url`: default is GRCh38. For GRCh37 datasets (e.g. denovo-db)
    pass `vep_url=VEP_URL_GRCH37`.
    """
    _ensure_dir(cache_path.parent)
    cached: dict[str, dict] = {}
    if cache_path.exists():
        cdf = pd.read_parquet(cache_path)
        cached = {r["variant_key"]: r.to_dict() for _, r in cdf.iterrows()}
    tokens = ext.apply(_variant_token, axis=1).tolist()
    keys = ext["variant_key"].tolist()
    need = [i for i, k in enumerate(keys) if k not in cached]
    if progress:
        print(
            f"  VEP annot: {len(cached):,} cached, {len(need):,} to fetch "
            f"[{vep_url.split('//', 1)[1].split('.', 1)[0]}]"
        )

    failed_batches = 0
    for start in range(0, len(need), BATCH_VEP):
        sub = need[start : start + BATCH_VEP]
        batch_tokens = [tokens[i] for i in sub]
        batch_keys = [keys[i] for i in sub]
        recs = _post_vep(batch_tokens, vep_url=vep_url)
        if not recs:
            # Transient VEP failure already logged inside _post_vep; leave these
            # variants out of the cache so the next run retries them.
            failed_batches += 1
        for vk, rec in zip(batch_keys, recs):
            t = _pick_canonical_missense(rec.get("transcript_consequences") or [])
            if t is None:
                cached[vk] = {
                    "variant_key": vk,
                    "transcript_id": None,
                    "protein_position": None,
                    "ref_aa": None,
                    "alt_aa": None,
                }
                continue
            aa = (t.get("amino_acids") or "").split("/")
            ref_aa = aa[0].strip().upper() if len(aa) == 2 else None
            alt_aa = aa[1].strip().upper() if len(aa) == 2 else None
            cached[vk] = {
                "variant_key": vk,
                "transcript_id": t.get("transcript_id"),
                "protein_position": int(t["protein_start"]) if t.get("protein_start") else None,
                "ref_aa": ref_aa,
                "alt_aa": alt_aa,
            }
        batch_idx = start // BATCH_VEP + 1
        if progress:
            done = start + len(sub)
            print(
                f"    VEP batch {batch_idx}/"
                f"{(len(need) + BATCH_VEP - 1) // BATCH_VEP}  "
                f"→ {done:,}/{len(need):,}"
            )
        # Checkpoint often — every 5 batches — so a session crash loses
        # at most 1,000 annotations.
        if batch_idx % 5 == 0:
            try:
                pd.DataFrame(list(cached.values())).to_parquet(cache_path, index=False)
            except Exception as exc:  # pragma: no cover - disk/IO errors
                print(f"    ⚠ cache write failed ({exc}); continuing without checkpoint")
        time.sleep(SLEEP)

    # Final flush — always write whatever we have, even after partial failures.
    try:
        pd.DataFrame(list(cached.values())).to_parquet(cache_path, index=False)
    except Exception as exc:  # pragma: no cover - disk/IO errors
        print(f"    ⚠ final cache write failed ({exc})")
    if failed_batches:
        print(
            f"  ⚠ {failed_batches} VEP batch(es) could not be retrieved this run "
            f"(~{failed_batches * BATCH_VEP:,} variants). Re-run the script "
            f"later and the cache-miss logic will pick them up automatically."
        )
    # Only return rows we actually have; callers merge back on variant_key.
    present = [k for k in keys if k in cached]
    return pd.DataFrame([cached[k] for k in present])


# ──────────────────────────── Sequence fetch ───────────────────────────


def fetch_sequences(
    transcript_ids: list[str],
    *,
    cache_path: Path,
    progress: bool = True,
    seq_url: str = SEQ_URL_DEFAULT,
) -> dict[str, str]:
    """Fetch canonical protein sequences from Ensembl /sequence/id. Cached to parquet."""
    _ensure_dir(cache_path.parent)
    cached: dict[str, str] = {}
    if cache_path.exists():
        cdf = pd.read_parquet(cache_path)
        cached = dict(zip(cdf["transcript_id"], cdf["sequence"]))
    need = [t for t in set(transcript_ids) if t and t not in cached]
    if progress:
        print(f"  seqs: {len(cached):,} cached, {len(need):,} to fetch")

    for i, tid in enumerate(need, 1):
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(
                    seq_url.format(tid=tid), headers={"Accept": "application/json"}, timeout=30
                )
                if r.status_code == 200:
                    cached[tid] = r.json().get("seq", "")
                    break
                if r.status_code in (429, 503):
                    time.sleep(2**attempt)
                    continue
                # 400 often = non-protein-coding transcript; record empty string so we don't retry.
                cached[tid] = ""
                break
            except requests.RequestException:
                time.sleep(1.5 * (attempt + 1))
        else:
            cached[tid] = ""
        if progress and (i % 50 == 0 or i == len(need)):
            print(f"    seq {i:,}/{len(need):,}")
        if i % 100 == 0 or i == len(need):
            pd.DataFrame(
                [{"transcript_id": k, "sequence": v} for k, v in cached.items()]
            ).to_parquet(cache_path, index=False)
        time.sleep(0.15)  # cheap per-request throttle

    return cached


# ──────────────────────────── ESM-2 scoring ────────────────────────────


@dataclass
class ESMRunner:
    model_name: str = ESM_MODEL
    device: str = "auto"
    max_len: int = 1022  # ESM-2 context window incl. CLS/EOS is 1024

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._torch = torch
        if self.device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device).eval()
        self.mask_id = self.tokenizer.mask_token_id
        # Map each single-letter AA to its token id (ESM tokenizer is char-level).
        self.aa_ids = {aa: self.tokenizer.convert_tokens_to_ids(aa) for aa in AA20}

    def _window(self, seq: str, position: int) -> tuple[str, int]:
        """If sequence > max_len, slice a window centered on the mutated position."""
        if len(seq) <= self.max_len:
            return seq, position
        half = self.max_len // 2
        start = max(0, position - 1 - half)
        end = min(len(seq), start + self.max_len)
        start = max(0, end - self.max_len)
        new_pos = (position - 1 - start) + 1  # re-1-index
        return seq[start:end], new_pos

    @staticmethod
    def _is_valid(ref: str, alt: str, seq: str, pos: int) -> bool:
        return (
            bool(seq)
            and 1 <= pos <= len(seq)
            and ref in AA20
            and alt in AA20
            and seq[pos - 1] == ref
        )

    def score_rows(self, rows: pd.DataFrame, seq_map: dict[str, str]) -> pd.DataFrame:
        """Score each row. `rows` must have
        `variant_key, transcript_id, protein_position, ref_aa, alt_aa`.
        Returns frame with LLR and probs; rows that can't be scored get NaN
        and a `skip_reason`.
        """
        torch = self._torch
        out_rows: list[dict] = []
        for i, r in enumerate(rows.itertuples(index=False), 1):
            vk = r.variant_key
            tid = r.transcript_id
            pos = r.protein_position
            ref = r.ref_aa
            alt = r.alt_aa
            base = {
                "variant_key": vk,
                "transcript_id": tid,
                "protein_position": pos,
                "ref_aa": ref,
                "alt_aa": alt,
                "esm2_prob_ref": np.nan,
                "esm2_prob_alt": np.nan,
                "esm2_llr": np.nan,
                "skip_reason": None,
            }
            seq = seq_map.get(tid, "")
            if not pos or pd.isna(pos) or not self._is_valid(ref, alt, seq, int(pos)):
                base["skip_reason"] = (
                    "no_seq"
                    if not seq
                    else (
                        "bad_pos"
                        if not pos or not (1 <= int(pos) <= len(seq))
                        else "ref_mismatch" if seq[int(pos) - 1] != ref else "non_canonical_aa"
                    )
                )
                out_rows.append(base)
                continue

            windowed_seq, windowed_pos = self._window(seq, int(pos))
            enc = self.tokenizer(windowed_seq, return_tensors="pt").to(self.device)
            # Position within input_ids: +1 for CLS.
            idx = windowed_pos  # 1-based aa pos + 1 offset for CLS = same value
            ids = enc.input_ids.clone()
            ids[0, idx] = self.mask_id
            with torch.no_grad():
                logits = self.model(ids, attention_mask=enc.attention_mask).logits
            probs = torch.softmax(logits[0, idx], dim=-1)
            p_ref = float(probs[self.aa_ids[ref]].item())
            p_alt = float(probs[self.aa_ids[alt]].item())
            base["esm2_prob_ref"] = p_ref
            base["esm2_prob_alt"] = p_alt
            # Guard against underflow: clip then take log-ratio.
            base["esm2_llr"] = math.log(max(p_alt, 1e-12)) - math.log(max(p_ref, 1e-12))
            out_rows.append(base)

            if i % 50 == 0 or i == len(rows):
                print(f"    ESM-2: {i:,}/{len(rows):,} scored")

        return pd.DataFrame(out_rows)


# ─────────────────────────────── Driver ────────────────────────────────


def score_variants(
    ext: pd.DataFrame,
    *,
    cache_dir: Path = CACHE_DIR,
    progress: bool = True,
    genome_build: str = "GRCh38",
) -> pd.DataFrame:
    """Top-level entry point: annotate → fetch seqs → ESM-2 score.

    `ext` must have `variant_key, chr, pos, ref, alt`. Returns a DataFrame
    with one row per input variant and the four ESM columns; variants that
    couldn't be resolved have `skip_reason` populated.

    `genome_build`: "GRCh38" (default, for our ClinVar-derived splits) or
    "GRCh37" (for denovo-db and other GRCh37 datasets).
    """
    if genome_build == "GRCh38":
        vep_url, seq_url = VEP_URL_DEFAULT, SEQ_URL_DEFAULT
    elif genome_build == "GRCh37":
        vep_url, seq_url = VEP_URL_GRCH37, SEQ_URL_GRCH37
    else:
        raise ValueError(f"genome_build must be GRCh37 or GRCh38; got {genome_build!r}")

    _ensure_dir(cache_dir)
    ann_path = cache_dir / "vep_ann.parquet"
    seq_path = cache_dir / "sequences.parquet"
    score_path = cache_dir / "scores.parquet"

    # 1. VEP annotation.
    if progress:
        print(f"[esm2] annotating {len(ext):,} variants via VEP REST ({genome_build})…")
    ann = annotate_with_vep(
        ext,
        cache_path=ann_path,
        progress=progress,
        vep_url=vep_url,
        seq_url=seq_url,
    )

    # 2. Unique transcripts → sequences.
    tids = list(ann["transcript_id"].dropna().unique())
    if progress:
        print(f"[esm2] fetching sequences for {len(tids):,} unique transcripts…")
    seqs = fetch_sequences(tids, cache_path=seq_path, progress=progress, seq_url=seq_url)

    # 3. Resume-aware ESM scoring.
    scored_cache: dict[str, dict] = {}
    if score_path.exists():
        sdf = pd.read_parquet(score_path)
        scored_cache = {r["variant_key"]: r.to_dict() for _, r in sdf.iterrows()}
    need = ann[~ann["variant_key"].isin(scored_cache)]
    if progress:
        print(f"[esm2] {len(scored_cache):,} cached scores, {len(need):,} to run through ESM-2…")

    if len(need) > 0:
        runner = ESMRunner()
        if progress:
            print(f"[esm2] loaded {runner.model_name} on {runner.device}")
        CHUNK = 500
        for chunk_start in range(0, len(need), CHUNK):
            chunk = need.iloc[chunk_start : chunk_start + CHUNK]
            new_scores = runner.score_rows(chunk, seqs)
            for _, row in new_scores.iterrows():
                scored_cache[row["variant_key"]] = row.to_dict()
            pd.DataFrame(list(scored_cache.values())).to_parquet(score_path, index=False)
            if progress:
                done = min(chunk_start + CHUNK, len(need))
                print(f"[esm2] checkpoint: {done:,}/{len(need):,} → {score_path.name}")

    return pd.DataFrame([scored_cache[k] for k in ann["variant_key"]])


def _load_denovo() -> pd.DataFrame:
    """Load denovo-db via the existing loader (no framework dep here)."""
    from src.external_validation.denovo_loader import load_denovo_db

    path = REPO / "data/raw/external/denovo_db/denovo-db.non-ssc-samples.variants.tsv.gz"
    df = load_denovo_db(path)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        choices=["denovo_db", "splits"],
        default="denovo_db",
        help="Which variant set to score",
    )
    ap.add_argument("--sample", type=int, default=0, help="Stratified subsample size (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/metrics/esm2_scores.parquet")
    args = ap.parse_args()

    if args.source == "denovo_db":
        ext = _load_denovo()[["variant_key", "chr", "pos", "ref", "alt", "label"]]
    else:
        parts = [
            pd.read_parquet(REPO / f"data/splits/{s}.parquet") for s in ("train", "val", "test")
        ]
        ext = pd.concat(parts)[
            ["variant_key", "chr", "pos", "ref", "alt", "label"]
        ].drop_duplicates("variant_key")

    if args.sample:
        per = args.sample // 2
        parts = []
        for lbl in (0, 1):
            sub = ext[ext["label"] == lbl]
            if len(sub):
                parts.append(
                    sub.sample(
                        min(per if lbl == 0 else args.sample - per, len(sub)),
                        random_state=args.seed,
                    )
                )
        ext = pd.concat(parts).reset_index(drop=True)
        print(f"[esm2] stratified sample → {len(ext):,} variants")

    scores = score_variants(ext)
    merged = ext.merge(scores, on="variant_key", how="left")
    out_path = REPO / args.out
    _ensure_dir(out_path.parent)
    merged.to_parquet(out_path, index=False)
    n_ok = merged["esm2_llr"].notna().sum()
    print(
        f"[esm2] wrote {out_path} — {n_ok:,}/{len(merged):,} variants scored "
        f"({n_ok/len(merged):.1%})"
    )


if __name__ == "__main__":
    main()
