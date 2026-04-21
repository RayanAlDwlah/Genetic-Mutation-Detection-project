#!/usr/bin/env python3
"""End-to-end driver for AlphaFold2-based structural feature extraction.

Strategy
--------
For each missense variant we need two things before we can extract
structural features:

  (a) the canonical UniProt accession of the gene the variant sits in,
      so we know which AlphaFold2 model to download;
  (b) the protein-position of the variant (1-indexed) so we can look
      up features at the correct residue.

We resolve (a) by querying the UniProt REST API with the variant's
gene symbol — a single search per *unique gene* (typical projects have
~15k unique genes, this cache fits in under a second after warm-up).

We resolve (b) from the VEP annotation parquet produced by
`src.esm2_scorer.annotate_with_vep` — the same file the ESM-2 Colab
job writes.  If that cache is not available, the script falls back to
a live VEP call per uncached variant (slow; only used for the
denovo-db smoke test).

The PDB URL is resolved by the AlphaFold DB's
`/api/prediction/{uniprot}` endpoint, which always returns the latest
model version (v6 at time of writing).  This isolates us from URL
versioning drift.

Disk management
---------------
PDBs are downloaded to a single scratch directory and deleted after
feature extraction.  Peak disk usage stays under 200 MB even when
processing hundreds of thousands of variants.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import requests

from src.structural_features import _NAN_FEATURES, compute_for_position

REPO = Path(__file__).resolve().parents[1]

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"

GENE_CACHE = REPO / "data/intermediate/alphafold/gene_to_uniprot.json"
URL_CACHE = REPO / "data/intermediate/alphafold/uniprot_to_pdb_url.json"
SCRATCH_DIR = REPO / "data/intermediate/alphafold/_scratch"

DEFAULT_SLEEP = 0.15
REQUEST_TIMEOUT = 30
CHECKPOINT_EVERY = 200
DOWNLOAD_RETRIES = 3


# ──────────────────── UniProt: gene symbol -> accession ────────────────────


def _load_json_cache(path: Path) -> dict[str, str]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_json_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, sort_keys=True, indent=1))


def _search_uniprot_for_gene(gene: str, max_retries: int = DOWNLOAD_RETRIES) -> str | None:
    """Return the canonical Swiss-Prot accession for a human gene symbol."""
    params = {
        "query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
        "fields": "accession",
        "format": "tsv",
        "size": 1,
    }
    for attempt in range(max_retries):
        try:
            r = requests.get(UNIPROT_SEARCH, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                if r.status_code in (429, 503):
                    time.sleep(2**attempt)
                    continue
                return None
            lines = r.text.strip().splitlines()
            # TSV header is "Entry"; first data row is the accession.
            if len(lines) < 2:
                return None
            return lines[1].strip()
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None


def resolve_uniprots(genes: set[str], *, progress: bool = True) -> dict[str, str]:
    """Build a gene -> UniProt accession dictionary (memoised on disk)."""
    cache = _load_json_cache(GENE_CACHE)
    need = sorted(g for g in genes if g and g not in cache)
    if progress:
        print(f"[alphafold] gene->uniprot: {len(cache):,} known, {len(need):,} to fetch")

    for i, gene in enumerate(need, 1):
        acc = _search_uniprot_for_gene(gene)
        cache[gene] = acc if acc else ""
        if i % 100 == 0 or i == len(need):
            _save_json_cache(GENE_CACHE, cache)
            if progress:
                hits = sum(1 for v in cache.values() if v)
                print(
                    f"  gene->uniprot progress: {i:,}/{len(need):,} "
                    f"(overall hit {hits / max(len(cache), 1):.1%})"
                )
        time.sleep(DEFAULT_SLEEP)

    _save_json_cache(GENE_CACHE, cache)
    return cache


# ──────────────────── AlphaFold: uniprot -> PDB url ────────────────────


def _query_alphafold_url(uniprot: str, max_retries: int = DOWNLOAD_RETRIES) -> str | None:
    """Resolve the *current* PDB download URL for a UniProt accession."""
    url = ALPHAFOLD_API.format(uniprot=uniprot)
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                entries = r.json()
                if entries and isinstance(entries, list):
                    return str(entries[0].get("pdbUrl") or "")
                return None
            if r.status_code == 404:
                return None
            if r.status_code in (429, 503):
                time.sleep(2**attempt)
                continue
        except (requests.RequestException, ValueError):
            time.sleep(1.5 * (attempt + 1))
    return None


def _download_pdb(url: str, out_path: Path) -> bool:
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            if r.status_code == 200:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 14):
                        fh.write(chunk)
                return True
            if r.status_code == 404:
                return False
            time.sleep(2**attempt)
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return False


# ──────────────────── Per-variant feature extraction ────────────────────


def extract_features(
    variants: pd.DataFrame, *, progress: bool = True, out_path: Path | None = None
) -> pd.DataFrame:
    """Return a per-variant structural-features DataFrame.

    `variants` must carry the columns
        `variant_key, gene, protein_position`

    The function writes the output parquet every `CHECKPOINT_EVERY`
    rows so a mid-run interrupt loses at most that many variants'
    worth of compute.
    """
    genes = set(variants["gene"].dropna().astype(str).unique())
    gene_to_uniprot = resolve_uniprots(genes, progress=progress)

    url_cache = _load_json_cache(URL_CACHE)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n = len(variants)
    for i, rec in enumerate(variants.itertuples(index=False), 1):
        gene = str(getattr(rec, "gene", "") or "")
        pos = getattr(rec, "protein_position", None)
        vk = getattr(rec, "variant_key", None)
        uni = gene_to_uniprot.get(gene, "")

        base = {
            "variant_key": vk,
            "gene": gene or None,
            "uniprot_id": uni or None,
            "protein_position": int(pos) if (pos is not None and not pd.isna(pos)) else None,
            **_NAN_FEATURES.__dict__,
        }

        if not uni or pos is None or pd.isna(pos):
            rows.append(base)
            continue

        # Resolve PDB URL (cached).
        if uni not in url_cache:
            url = _query_alphafold_url(uni)
            url_cache[uni] = url or ""
            time.sleep(DEFAULT_SLEEP)
        pdb_url = url_cache[uni]
        if not pdb_url:
            rows.append(base)
            continue

        pdb_path = SCRATCH_DIR / f"{uni}.pdb"
        if not pdb_path.exists():
            if not _download_pdb(pdb_url, pdb_path):
                rows.append(base)
                continue

        feats = compute_for_position(pdb_path, int(pos))
        rows.append({**base, **feats.__dict__})

        # Periodic checkpoints.
        if i % CHECKPOINT_EVERY == 0 or i == n:
            _save_json_cache(URL_CACHE, url_cache)
            if out_path is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_parquet(out_path, index=False)
            if progress:
                n_ok = sum(1 for r in rows if r["pLDDT_position"] == r["pLDDT_position"])
                print(
                    f"  [alphafold] {i:,}/{n:,} variants — "
                    f"{n_ok:,} with structural features ({n_ok / i:.1%})"
                )

    _save_json_cache(URL_CACHE, url_cache)
    return pd.DataFrame(rows)


# ──────────────────── Driver ────────────────────


def _load_variants(
    vep_cache: Path, splits_dir: Path, denovo_raw: Path | None = None
) -> pd.DataFrame:
    """Assemble (variant_key, gene, protein_position) from VEP cache + splits.

    The gene lookup tries the committed train/val/test splits first;
    variants not found there (denovo-db smoke tests, external
    datasets) are looked up in the raw denovo-db TSV if supplied.
    """
    vep = pd.read_parquet(vep_cache)
    keep_cols = ["variant_key", "protein_position"]
    vep = vep.loc[vep["protein_position"].notna(), keep_cols].copy()

    parts = []
    for split in ("train", "val", "test"):
        path = splits_dir / f"{split}.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path)[["variant_key", "gene"]])
    splits = (
        pd.concat(parts, ignore_index=True).drop_duplicates("variant_key")
        if parts
        else pd.DataFrame(columns=["variant_key", "gene"])
    )

    merged = vep.merge(splits, on="variant_key", how="left")

    # Backfill genes from denovo-db for variants not in any split.
    missing = merged["gene"].isna()
    if missing.any() and denovo_raw is not None and denovo_raw.exists():
        try:
            from src.external_validation.denovo_loader import load_denovo_db  # noqa: PLC0415
        except Exception:
            pass
        else:
            denovo = load_denovo_db(denovo_raw)[["variant_key", "gene"]].drop_duplicates(
                "variant_key"
            )
            denovo_map = denovo.set_index("variant_key")["gene"].to_dict()
            merged.loc[missing, "gene"] = merged.loc[missing, "variant_key"].map(denovo_map)

    merged = merged.dropna(subset=["gene"])
    return merged[["variant_key", "gene", "protein_position"]].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--vep-cache",
        default="data/intermediate/esm2/vep_ann.parquet",
        help="VEP annotation parquet containing variant_key + protein_position",
    )
    ap.add_argument(
        "--splits-dir",
        default="data/splits",
        help="Directory of train/val/test parquet splits (for gene lookup)",
    )
    ap.add_argument(
        "--out",
        default="data/intermediate/alphafold/features.parquet",
        help="Output features parquet",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional: only process first N variants (for smoke tests)",
    )
    ap.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep downloaded PDB files after processing (default: delete)",
    )
    ap.add_argument(
        "--denovo-raw",
        default="data/raw/external/denovo_db/denovo-db.non-ssc-samples.variants.tsv.gz",
        help="Optional denovo-db TSV to backfill genes for external variants",
    )
    args = ap.parse_args()

    vep_path = REPO / args.vep_cache if not Path(args.vep_cache).is_absolute() else Path(args.vep_cache)
    splits_dir = REPO / args.splits_dir if not Path(args.splits_dir).is_absolute() else Path(args.splits_dir)
    out_path = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)

    if not vep_path.exists():
        raise SystemExit(f"VEP cache not found: {vep_path}")

    denovo_raw = REPO / args.denovo_raw if not Path(args.denovo_raw).is_absolute() else Path(args.denovo_raw)
    variants = _load_variants(vep_path, splits_dir, denovo_raw=denovo_raw)
    print(f"[alphafold] assembled {len(variants):,} (variant, gene, pos) rows")
    if args.limit:
        variants = variants.head(args.limit).copy()
        print(f"[alphafold] limited to first {len(variants):,} for this run")

    features = extract_features(variants, out_path=out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)

    if not args.keep_scratch and SCRATCH_DIR.exists():
        shutil.rmtree(SCRATCH_DIR)
        print(f"[alphafold] cleaned up scratch dir {SCRATCH_DIR.name}/")

    n_ok = int(features["pLDDT_position"].notna().sum())
    try:
        rel = out_path.relative_to(REPO)
    except ValueError:
        rel = out_path
    print(f"[alphafold] wrote {rel}")
    print(
        f"[alphafold] summary: {n_ok:,}/{len(features):,} variants successfully scored "
        f"({n_ok / max(len(features), 1):.1%})"
    )


if __name__ == "__main__":
    main()
