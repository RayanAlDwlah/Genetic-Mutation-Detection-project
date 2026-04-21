# Makefile — canonical entry points for every reproducible step.
#
# Design goals:
#   1. Every target is idempotent: rerunning produces the same artifacts.
#   2. Every target works identically on host (PYTHONPATH=.) and in Docker.
#   3. README "how to reproduce" commands are literally these targets.

.PHONY: help install test lint format fix typecheck \
        train evaluate external ablate-esm2 reproduce-headline \
        verify-leakage report \
        docker-build docker-test docker-reproduce \
        clean-artifacts

PYTHON ?= python
PYTHONPATH := $(CURDIR)
export PYTHONPATH

## help:  show this help
help:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

## install:  install dev-grade dependencies from the exact-pinned lock file
install:
	$(PYTHON) -m pip install -r requirements-lock.txt

## test:  run pytest with coverage gate (≥ 55%)
test:
	$(PYTHON) -m pytest tests/ -n auto

## lint:  ruff + black --check (no edits)
lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/
	$(PYTHON) -m black --check src/ tests/ scripts/

## format:  auto-apply ruff + black (edits files)
format:
	$(PYTHON) -m ruff check --fix src/ tests/ scripts/
	$(PYTHON) -m black src/ tests/ scripts/

## fix:  alias for format
fix: format

## typecheck:  mypy on the actively-maintained source tree
typecheck:
	$(PYTHON) -m mypy src/

## verify-leakage:  CLI leakage gate (same checks wrapped in pytest)
verify-leakage:
	$(PYTHON) -m src.verify_no_leakage

## train:  retrain the XGBoost baseline (~5–10 min)
train:
	$(PYTHON) -m src.training --trials 40 --seed 42

## evaluate:  rebuild bootstrap CIs / calibration / operating points
evaluate:
	$(PYTHON) -m src.evaluate_baseline

## external:  re-score denovo-db external validation (requires VEP REST)
external:
	$(PYTHON) scripts/evaluate_external.py --only denovo_db --use-vep --sample 1000 --n-boot 1000

## ablate-esm2:  retrain with/without esm2_llr feature, diff the headlines
ablate-esm2:
	$(PYTHON) scripts/ablate_esm2.py

## reproduce-headline:  integration test that rebinds the committed metrics to
##                      the committed checkpoint (runs in < 30 s).
reproduce-headline:
	$(PYTHON) -m pytest tests/integration/test_reproduce_headline.py -v --no-cov

## report:  build the LaTeX technical report (requires pdflatex + bibtex)
report:
	@if ! command -v pdflatex >/dev/null 2>&1; then \
	    echo "ERROR: pdflatex not installed."; \
	    echo "Install with: brew install --cask mactex  (or use Overleaf)"; \
	    exit 1; \
	fi
	@cd report && \
	    cp ../results/figures/{leakage_journey,calibration_triptych,shap_summary,shap_bar,baselines_forest_plot}.png figures/ 2>/dev/null || true && \
	    pdflatex -interaction=nonstopmode main.tex && \
	    bibtex main && \
	    pdflatex -interaction=nonstopmode main.tex && \
	    pdflatex -interaction=nonstopmode main.tex && \
	    echo "Built report/main.pdf"

# ───────────────────────── Docker shortcuts ─────────────────────────

## docker-build:  build the reproducibility image (tagged missense-classifier)
docker-build:
	docker build -t missense-classifier .

## docker-test:  pytest + leakage gate inside the container
docker-test: docker-build
	docker run --rm missense-classifier make test verify-leakage

## docker-reproduce:  end-to-end repro check inside a clean container
docker-reproduce: docker-build
	docker run --rm missense-classifier make reproduce-headline

# ────────────────────────── Cleanup helpers ──────────────────────────

## clean-artifacts:  wipe regeneratable results/ files (keeps raw data intact)
clean-artifacts:
	@echo "Removing regeneratable artifacts…"
	rm -rf results/metrics/xgboost_*.csv \
	       results/metrics/xgboost_predictions.parquet \
	       results/metrics/baselines_comparison.csv \
	       results/figures/xgboost_* \
	       coverage.xml \
	       .pytest_cache \
	       .mypy_cache \
	       .ruff_cache
	@echo "Done. Rerun \`make train && make evaluate\` to rebuild."
