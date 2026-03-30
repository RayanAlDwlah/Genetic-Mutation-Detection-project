# Data Layout

This repository tracks the lightweight parquet snapshots needed for collaboration and quick
reproducibility.

Included in git:
- `intermediate/clinvar_labeled_clean.parquet`
- `intermediate/dbnsfp_selected_features.parquet`
- `intermediate/gnomad_af_clean.parquet`
- `processed/merged_clinvar_gnomad_dbnsfp.parquet`
- `processed/final_balanced.parquet`
- `processed/final_strict.parquet`
- `splits/train.parquet`
- `splits/val.parquet`
- `splits/test.parquet`
- `splits/strict/train.parquet`
- `splits/strict/val.parquet`
- `splits/strict/test.parquet`

Not included in git:
- `raw/` source downloads such as ClinVar, gnomAD, dbNSFP, and UniProt

Quick run after cloning:

```bash
python -m src.training
```
