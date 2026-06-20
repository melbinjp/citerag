"""
backend/ingestion/run.py

Repeatable corpus ingestion procedure.

Walks a configured CORPUS_DIR for all ``*.pdf`` files, runs each through the
full IngestionPipeline, persists the structured error log, and returns a
summary of the run.

Design highlights
-----------------
- **Idempotent**: re-running against the same source set produces no
  duplicates because the VectorStore uses a deterministic SHA-256 point id
  and the pipeline checks ``VectorStore.exists()`` before embedding (Req 4.4,
  5.4).
- **Continue-on-failure**: a document-level failure is recorded in the error
  log and counted toward ``failed_filenames``; the remaining PDFs are always
  processed (Req 1.7, 5.5).
- **Scale**: supports 10–100 PDFs each containing 200–2000 pages (Req 5.1,
  5.2).
- **Reporting**: returns the count of successfully ingested PDFs and the
  filenames of any that failed (Req 5.6).
- **Importable**: all logic lives in :func:`run_corpus_ingestion` so it can
  be called from other modules or tests without executing ``main()``.

Requirements: 5.1, 5.2, 5.4, 5.6
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CorpusIngestionRun:
    """Summary produced by :func:`run_corpus_ingestion`.

    Attributes
    ----------
    total_pdfs:
        Total number of PDF files discovered in ``corpus_dir``.
    ingested_count:
        Number of PDFs that were processed without a document-level failure
        (i.e. ``IngestionResult.success is True``).
    failed_filenames:
        Basenames of PDF files that experienced a document-level failure
        (corrupt, encrypted, zero-text, or unexpected error).
    error_log_path:
        Filesystem path where the error log was flushed, or ``None`` when no
        errors were recorded and no flush occurred.
    """

    total_pdfs: int
    ingested_count: int
    failed_filenames: list[str]
    error_log_path: Optional[str]


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def run_corpus_ingestion(
    corpus_dir: str,
    error_log_path: str,
    qdrant_url: str,
    collection_name: str,
) -> CorpusIngestionRun:
    """Walk *corpus_dir*, ingest every PDF, flush errors, and return a summary.

    Parameters
    ----------
    corpus_dir:
        Directory (searched recursively) that contains the PDF corpus.
    error_log_path:
        Destination file for the JSON Lines error log.  Errors are *appended*
        to this file so that multiple partial runs accumulate without losing
        earlier failures.
    qdrant_url:
        Qdrant server URL (e.g. ``"http://qdrant:6333"``).
    collection_name:
        Qdrant collection name (e.g. ``"corpus"``).

    Returns
    -------
    CorpusIngestionRun
        Summary with total PDF count, successfully ingested count, failed
        filenames, and the path where the error log was written (if any errors
        occurred).

    Raises
    ------
    FileNotFoundError
        If *corpus_dir* does not exist or is not a directory.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"CORPUS_DIR '{corpus_dir}' does not exist."
        )
    if not corpus_path.is_dir():
        raise FileNotFoundError(
            f"CORPUS_DIR '{corpus_dir}' is not a directory."
        )

    # ── Discover PDFs ────────────────────────────────────────────────────────
    # Recursive glob; sort for deterministic processing order across runs.
    pdf_files: list[Path] = sorted(corpus_path.rglob("*.pdf"))
    total_pdfs = len(pdf_files)

    logger.info(
        "corpus_ingestion.start corpus_dir=%s total_pdfs=%d",
        corpus_dir,
        total_pdfs,
    )

    if total_pdfs == 0:
        logger.warning(
            "corpus_ingestion.no_pdfs corpus_dir=%s", corpus_dir
        )
        return CorpusIngestionRun(
            total_pdfs=0,
            ingested_count=0,
            failed_filenames=[],
            error_log_path=None,
        )

    # ── Build pipeline components ────────────────────────────────────────────
    # Each component is constructed once and reused for all PDFs in the run.
    # This avoids reloading the (large) embedding model on every document.
    try:
        from backend.ingestion.extract import Text_Extractor
        from backend.ingestion.clean import Cleaner
        from backend.ingestion.chunker import Chunker, ChunkerConfig
        from backend.ingestion.language import LanguageDetector
        from backend.ingestion.error_log import IngestionErrorLog
        from backend.ingestion.pipeline import IngestionPipeline
        from backend.embedding import EmbeddingModel
        from backend.vector_store import VectorStore
    except ImportError:
        # Fallback for environments where sys.path points at the backend dir
        from ingestion.extract import Text_Extractor  # type: ignore[no-redef]
        from ingestion.clean import Cleaner  # type: ignore[no-redef]
        from ingestion.chunker import Chunker, ChunkerConfig  # type: ignore[no-redef]
        from ingestion.language import LanguageDetector  # type: ignore[no-redef]
        from ingestion.error_log import IngestionErrorLog  # type: ignore[no-redef]
        from ingestion.pipeline import IngestionPipeline  # type: ignore[no-redef]
        from embedding import EmbeddingModel  # type: ignore[no-redef]
        from vector_store import VectorStore  # type: ignore[no-redef]

    logger.info("corpus_ingestion.loading_components")
    extractor = Text_Extractor()
    cleaner = Cleaner()
    chunker = Chunker(config=ChunkerConfig())
    error_log = IngestionErrorLog()

    logger.info("corpus_ingestion.loading_embedding_model")
    embedding_model = EmbeddingModel()

    logger.info("corpus_ingestion.connecting_vector_store url=%s collection=%s",
                qdrant_url, collection_name)
    vector_store = VectorStore(
        url=qdrant_url,
        collection_name=collection_name,
        auto_ensure=True,
    )

    pipeline = IngestionPipeline(
        extractor=extractor,
        cleaner=cleaner,
        chunker=chunker,
        embedding_model=embedding_model,
        vector_store=vector_store,
        error_log=error_log,
    )

    # ── Process each PDF ─────────────────────────────────────────────────────
    ingested_count = 0
    failed_filenames: list[str] = []

    for idx, pdf_path in enumerate(pdf_files, start=1):
        pdf_str = str(pdf_path)
        filename = pdf_path.name

        logger.info(
            "corpus_ingestion.ingesting [%d/%d] filename=%s",
            idx,
            total_pdfs,
            filename,
        )

        try:
            result = pipeline.ingest_pdf(pdf_str)
        except Exception as exc:  # noqa: BLE001 — unexpected pipeline-level error
            logger.error(
                "corpus_ingestion.unexpected_error filename=%s reason=%s",
                filename,
                exc,
            )
            failed_filenames.append(filename)
            continue

        if result.success:
            ingested_count += 1
            logger.info(
                "corpus_ingestion.success filename=%s chunks=%d",
                filename,
                result.num_chunks,
            )
        else:
            failed_filenames.append(filename)
            logger.warning(
                "corpus_ingestion.failed filename=%s errors=%d",
                filename,
                len(result.errors),
            )

    # ── Flush error log ──────────────────────────────────────────────────────
    # Req 5.6 / 13.5: persist all accumulated error entries to durable storage.
    flushed_log_path: Optional[str] = None
    if len(error_log) > 0:
        error_log.flush(error_log_path)
        flushed_log_path = error_log_path
        logger.info(
            "corpus_ingestion.error_log_flushed path=%s entries=%d",
            error_log_path,
            len(error_log),
        )

    # ── Report (Req 5.6) ─────────────────────────────────────────────────────
    logger.info(
        "corpus_ingestion.complete total=%d ingested=%d failed=%d",
        total_pdfs,
        ingested_count,
        len(failed_filenames),
    )

    return CorpusIngestionRun(
        total_pdfs=total_pdfs,
        ingested_count=ingested_count,
        failed_filenames=failed_filenames,
        error_log_path=flushed_log_path,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Read configuration from environment variables and run the ingestion.

    Environment variables
    ---------------------
    CORPUS_DIR
        Directory containing the PDF corpus (default: ``./data/corpus``).
    INGESTION_ERROR_LOG
        Path for the JSON Lines error log (default: ``./data/ingestion_errors.jsonl``).
    QDRANT_URL
        Qdrant server URL (default: ``http://qdrant:6333``).
    QDRANT_COLLECTION
        Qdrant collection name (default: ``corpus``).

    Exit codes
    ----------
    0 : All PDFs ingested successfully (or no PDFs found).
    1 : One or more PDFs failed.
    """
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    corpus_dir = os.environ.get("CORPUS_DIR", "./data/corpus")
    error_log_path = os.environ.get("INGESTION_ERROR_LOG", "./data/ingestion_errors.jsonl")
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    collection_name = os.environ.get("QDRANT_COLLECTION", "corpus")

    logger.info(
        "corpus_ingestion.config corpus_dir=%s error_log=%s qdrant_url=%s collection=%s",
        corpus_dir,
        error_log_path,
        qdrant_url,
        collection_name,
    )

    try:
        run = run_corpus_ingestion(
            corpus_dir=corpus_dir,
            error_log_path=error_log_path,
            qdrant_url=qdrant_url,
            collection_name=collection_name,
        )
    except FileNotFoundError as exc:
        logger.error("corpus_ingestion.fatal %s", exc)
        sys.exit(1)

    # ── Human-readable summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Corpus Ingestion Summary")
    print(f"{'='*60}")
    print(f"  Total PDFs found   : {run.total_pdfs}")
    print(f"  Successfully ingested: {run.ingested_count}")
    print(f"  Failed             : {len(run.failed_filenames)}")

    if run.failed_filenames:
        print(f"\n  Failed filenames:")
        for name in run.failed_filenames:
            print(f"    - {name}")

    if run.error_log_path:
        print(f"\n  Error log written to: {run.error_log_path}")
    else:
        print(f"\n  No errors recorded.")

    print(f"{'='*60}\n")

    sys.exit(0 if not run.failed_filenames else 1)


if __name__ == "__main__":
    main()
