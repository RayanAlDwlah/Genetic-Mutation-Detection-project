# Reproducibility container for the paralog-aware, externally-validated
# missense classifier. `docker build -t missense-classifier .` produces
# an image that passes `make test && make reproduce-headline` identically
# to the host developer machine.

FROM python:3.11.7-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/RayanAlDwlah/Genetic-Mutation-Detection-project"
LABEL org.opencontainers.image.description="XGBoost + gnomAD constraint + ESM-2 missense classifier with paralog-aware evaluation."
LABEL org.opencontainers.image.licenses="MIT"

# System dependencies:
#  - liblzma-dev is the Achilles heel of the host pyenv build we hit;
#    pinning it here means ESM-2's HF tokenizer loads on first try.
#  - dssp is the Kabsch-Sander secondary-structure assigner used by our
#    structural-feature extractor (Stage 2.2).
#  - libopenblas is faster than the default BLAS for numpy/scipy; the
#    image loses ~1 min off training and ~15 s off every pytest run.
#  - build-essential is temporary (needed by biopython wheels); removed
#    in the same RUN layer so it doesn't bloat the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        liblzma-dev \
        dssp \
        libopenblas-dev \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# Layer cache friendly: dependencies first, then source. Re-run of
# `docker build` after a code change skips the dep install.
COPY requirements.txt requirements-lock.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-lock.txt

# Remove build tools to slim the final image (keeps runtime < 2 GB).
RUN apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/* /root/.cache

COPY . .

# Sanity: confirm the binary + tokenizer stack loads at build time, not
# runtime. If HF ever ships a bad wheel, the build fails immediately.
RUN python -c "import lzma; from transformers import AutoTokenizer; print('tokenizer stack OK')"

CMD ["make", "help"]
