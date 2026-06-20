FROM python:3.10-slim

# The application code lives under /app/backend so that `backend` is an
# importable package (the RAG modules use `from backend.x import ...`) while
# legacy sibling imports (`utils`, `rag_session`, `user_session`) still resolve.
# PYTHONPATH includes both /app (for `backend.*`) and /app/backend (for siblings).
WORKDIR /app/backend

# Install system dependencies: tesseract-ocr for OCR support (Req 1.2, 1.3)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install torch (CPU-only) first so subsequent deps resolve against it (Req 14.2, 14.3)
COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face model cache in a writable directory
RUN mkdir -p /tmp/hf_cache && chmod -R 777 /tmp/hf_cache
ENV HF_HOME=/tmp/hf_cache
ENV HF_HUB_DOWNLOAD_TIMEOUT=120

# Make both import styles work:
#   - `from backend.embedding import ...`  → resolved via /app
#   - `from utils.loaders import ...`       → resolved via /app/backend
ENV PYTHONPATH=/app:/app/backend
ENV PYTHONUNBUFFERED=1

# Copy the application code into the package directory
COPY . /app/backend/

# Corpus and error-log directories (created here so the volume mount works even
# if the host directory does not exist yet)
RUN mkdir -p /data/corpus

EXPOSE 7860

# Entrypoint: optionally ingest corpus before starting the API server.
# If CORPUS_DIR contains PDF files, the env-driven ingestion runner is executed
# first (idempotent — already-ingested chunks are skipped via the point-id
# check in VectorStore). If SKIP_INGESTION=true the step is bypassed so that
# containers sharing a pre-populated Qdrant volume start faster. (Req 14.1, 14.5)
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
