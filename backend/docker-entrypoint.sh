#!/bin/sh
# docker-entrypoint.sh
# ---------------------------------------------------------------------------
# Optional corpus ingestion before starting the API server.
#
# Environment variables:
#   CORPUS_DIR          Path to the directory containing PDF files to ingest.
#                       Defaults to /data/corpus.
#   SKIP_INGESTION      Set to "true" to bypass ingestion and start the server
#                       immediately (e.g. when the Qdrant volume is already
#                       populated).  Defaults to "false".
#   INGESTION_ERROR_LOG Path for the JSON-Lines ingestion error log.
#                       Defaults to /data/ingestion_errors.jsonl.
#   QDRANT_URL          Qdrant server URL (default http://qdrant:6333).
#   QDRANT_COLLECTION   Qdrant collection name (default corpus).
# ---------------------------------------------------------------------------

set -e

CORPUS_DIR="${CORPUS_DIR:-/data/corpus}"
SKIP_INGESTION="${SKIP_INGESTION:-false}"
INGESTION_ERROR_LOG="${INGESTION_ERROR_LOG:-/data/ingestion_errors.jsonl}"
QDRANT_URL="${QDRANT_URL-http://qdrant:6333}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-corpus}"

export CORPUS_DIR INGESTION_ERROR_LOG QDRANT_URL QDRANT_COLLECTION

if [ "$SKIP_INGESTION" = "true" ]; then
    echo "[entrypoint] SKIP_INGESTION=true — skipping corpus ingestion."
else
    # Count PDF files in CORPUS_DIR (recursively).
    PDF_COUNT=0
    if [ -d "$CORPUS_DIR" ]; then
        PDF_COUNT=$(find "$CORPUS_DIR" -iname "*.pdf" | wc -l)
    fi

    if [ "$PDF_COUNT" -gt 0 ]; then
        echo "[entrypoint] Found $PDF_COUNT PDF(s) in $CORPUS_DIR — running ingestion..."
        # The runner reads CORPUS_DIR / INGESTION_ERROR_LOG / QDRANT_URL /
        # QDRANT_COLLECTION from the environment. It is idempotent: chunks whose
        # point id already exists in Qdrant are skipped without re-embedding.
        python -m backend.ingestion.run \
            || echo "[entrypoint] WARNING: ingestion exited with errors — check $INGESTION_ERROR_LOG"
        echo "[entrypoint] Ingestion complete."
    else
        echo "[entrypoint] No PDF files found in $CORPUS_DIR — skipping ingestion."
    fi
fi

echo "[entrypoint] Starting RAG API server..."
exec uvicorn app:app --host 0.0.0.0 --port 7860
