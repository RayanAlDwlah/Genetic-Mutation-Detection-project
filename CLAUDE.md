# AI Agent Instructions

## Project Goal
Build and evaluate models for genetic mutation detection using tabular and sequence-based features.

## Repository Map
- `data/raw/`: untouched source data.
- `data/processed/`: cleaned data used by training jobs.
- `data/splits/`: train/val/test split artifacts.
- `notebooks/`: step-by-step experimentation notebooks (01-06).
- `src/`: production Python modules for data, features, models, training, and evaluation.
- `configs/config.yaml`: central runtime configuration.
- `results/figures/`: plots and visual diagnostics.
- `results/checkpoints/`: model checkpoints.
- `results/metrics/`: evaluation outputs.

## Working Rules
1. Prefer implementing reusable logic in `src/` instead of notebooks.
2. Keep notebooks focused on exploration and reporting.
3. Do not commit raw datasets or large generated files.
4. Update `configs/config.yaml` when adding new runtime parameters.
5. Save evaluation outputs under `results/metrics/` with clear names and timestamps.

## Coding Conventions
- Use Python 3.10+ style and type hints where practical.
- Write small, testable functions with clear docstrings.
- Keep model interfaces consistent (`fit`, `predict`, optional `predict_proba`).
- Avoid hardcoded paths; use config-driven paths.

## Agent Task Priorities
1. Data integrity and reproducibility.
2. Baseline model stability before deep model complexity.
3. Clear experiment tracking (config + metrics + checkpoint).
4. Maintainable code over notebook-only implementations.
