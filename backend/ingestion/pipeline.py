"""
backend/ingestion/pipeline.py

Ingestion_Pipeline: orchestrates the full PDF ingestion flow.

Pipeline stages for each PDF:
    1. Open with PyMuPDF (Text_Extractor) — on failure: log error, skip doc, continue
    2. For each page: extract native text / OCR
    3. Clean + normalize whitespace + remove header/footer (Cleaner)
    4. Detect language (Language_Detector)
    5. Recursive chunking with metadata (Chunker)
    6. For each chunk:
        a. If chunk already exists in VectorStore (sha256 match): skip embedding
        b. Else: embed with bge-m3 — on failure: log error, continue remaining chunks
        c. Upsert to Qdrant — on failure: log error, continue remaining chunks
    7. Return IngestionResult with counts and errors

Continue-on-failure semantics:
    - Exactly one structured error entry is logged per failure (Req 13.4)
    - A failure at any stage never aborts the remaining work (Req 1.6, 1.7, 4.6, 4.7, 5.5)
    - Previously persisted chunks are left unchanged on persistence failure (Req 4.7)

Idempotency:
    - VectorStore.exists(point_id) is checked before each embed call (Req 4.4)
    - Re-ingesting an identical chunk is a no-op (no new embedding, no duplicate)

Requirements: 1.6, 1.7, 4.4, 4.6, 4.7, 5.3, 5.5, 13.4
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

try:
    # When running as part of the installed 'backend' package
    from backend.ingestion.error_log import IngestionErrorEntry, IngestionErrorLog
    from backend.ingestion.extract import ExtractionError
except ImportError:
    # When sys.path points directly at the backend directory (test runner mode)
    from ingestion.error_log import IngestionErrorEntry, IngestionErrorLog  # type: ignore[no-redef]
    from ingestion.extract import ExtractionError  # type: ignore[no-redef]

if TYPE_CHECKING:
    from backend.ingestion.chunker import Chunker, PageText as ChunkerPageText
    from backend.ingestion.clean import Cleaner
    from backend.ingestion.extract import Text_Extractor
    from backend.ingestion.language import LanguageDetector
    from backend.embedding import EmbeddingModel
    from backend.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class IngestionResult:
    """Summary of a single PDF ingestion run.

    Attributes:
        pdf_path:   Absolute or relative path to the ingested PDF.
        num_chunks: Number of chunks successfully upserted to the VectorStore.
        errors:     All structured error entries recorded during this run.
        success:    True when no document-level failure occurred (extraction
                    succeeded and at least processing started); False when the
                    document was skipped entirely (e.g. corrupt/encrypted PDF).
    """

    pdf_path: str
    num_chunks: int
    errors: list[IngestionErrorEntry]
    success: bool


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """Orchestrates extraction → cleaning → language detection → chunking →
    embedding → idempotent upsert for one PDF at a time.

    All stages use continue-on-failure semantics: a failure at any stage is
    logged as exactly one structured ``IngestionErrorEntry`` and processing
    continues with the next unit of work (next page, next chunk, next PDF).

    Parameters
    ----------
    extractor:
        A ``Text_Extractor``-compatible object exposing ``extract(pdf_path)``.
    cleaner:
        A ``Cleaner``-compatible object exposing ``clean_document(pages)``.
    chunker:
        A ``Chunker``-compatible object exposing ``chunk(pages)``.
    embedding_model:
        An ``EmbeddingModel``-compatible object exposing ``embed(texts)``.
    vector_store:
        A ``VectorStore``-compatible object exposing ``exists``, ``upsert_chunk``,
        and ``compute_point_id``.
    error_log:
        An ``IngestionErrorLog`` instance that receives error entries.
    """

    def __init__(
        self,
        extractor: "Text_Extractor",
        cleaner: "Cleaner",
        chunker: "Chunker",
        embedding_model: "EmbeddingModel",
        vector_store: "VectorStore",
        error_log: IngestionErrorLog,
    ) -> None:
        self._extractor = extractor
        self._cleaner = cleaner
        self._chunker = chunker
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._error_log = error_log

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_pdf(self, pdf_path: str) -> IngestionResult:
        """Run the full pipeline for one PDF with continue-on-failure semantics.

        Parameters
        ----------
        pdf_path:
            Filesystem path to the PDF to ingest.

        Returns
        -------
        IngestionResult
            Summary containing the number of successfully upserted chunks,
            all error entries logged during this run, and a ``success`` flag
            indicating whether document-level processing could begin.
        """
        filename = os.path.basename(pdf_path)
        run_errors: list[IngestionErrorEntry] = []
        num_chunks_upserted = 0

        # ── Stage 1: Extract pages (Req 1.7 — corrupt/encrypted → log + skip) ──
        try:
            extracted_pages = self._extractor.extract(pdf_path)
        except ExtractionError as exc:
            entry = IngestionErrorEntry.now(
                filename=filename,
                stage="extraction",
                reason=str(exc),
                page_number=None,
            )
            self._error_log.append(entry)
            run_errors.append(entry)
            logger.error(
                "ingestion.extraction filename=%s reason=%s",
                filename,
                exc,
            )
            return IngestionResult(
                pdf_path=pdf_path,
                num_chunks=0,
                errors=run_errors,
                success=False,
            )
        except Exception as exc:  # noqa: BLE001 — unexpected errors also skip doc
            entry = IngestionErrorEntry.now(
                filename=filename,
                stage="extraction",
                reason=f"Unexpected error: {exc}",
                page_number=None,
            )
            self._error_log.append(entry)
            run_errors.append(entry)
            logger.error(
                "ingestion.extraction filename=%s reason=unexpected: %s",
                filename,
                exc,
            )
            return IngestionResult(
                pdf_path=pdf_path,
                num_chunks=0,
                errors=run_errors,
                success=False,
            )

        # ── Stage 2: Check for zero-text document (Req 1.6, 5.5) ─────────────
        # A page with empty text is retained in extracted_pages by the extractor
        # (so the caller can detect it); we must check whether *all* pages yielded
        # no text after extraction.
        all_page_texts = [p.text for p in extracted_pages]
        total_chars = sum(len(t.strip()) for t in all_page_texts)

        if total_chars == 0 and extracted_pages:
            entry = IngestionErrorEntry.now(
                filename=filename,
                stage="extraction",
                reason="All pages yielded zero characters after extraction and OCR",
                page_number=None,
            )
            self._error_log.append(entry)
            run_errors.append(entry)
            logger.error(
                "ingestion.extraction filename=%s reason=zero total chars",
                filename,
            )
            # Continue to next PDF (success=False but not a doc-open failure)
            return IngestionResult(
                pdf_path=pdf_path,
                num_chunks=0,
                errors=run_errors,
                success=False,
            )

        # ── Stage 3: Clean text (Req 2.1, 2.2, 2.3) ─────────────────────────
        raw_texts = [p.text for p in extracted_pages]
        try:
            cleaned_texts = self._cleaner.clean_document(raw_texts)
        except Exception as exc:  # noqa: BLE001
            # Cleaning failure is non-fatal; fall back to raw texts
            entry = IngestionErrorEntry.now(
                filename=filename,
                stage="extraction",
                reason=f"Text cleaning failed: {exc}",
                page_number=None,
            )
            self._error_log.append(entry)
            run_errors.append(entry)
            logger.error(
                "ingestion.cleaning filename=%s reason=%s",
                filename,
                exc,
            )
            cleaned_texts = raw_texts  # best-effort fallback

        # ── Stage 4: Language detection & build ChunkerPageText objects ──────
        # Import here to avoid circular imports at module level
        try:
            from backend.ingestion.chunker import PageText as ChunkerPageText  # noqa: PLC0415
        except ImportError:
            from ingestion.chunker import PageText as ChunkerPageText  # type: ignore[no-redef]  # noqa: PLC0415

        chunker_pages: list[ChunkerPageText] = []
        pdf_id = self._derive_pdf_id(pdf_path)

        for orig_page, cleaned_text in zip(extracted_pages, cleaned_texts):
            # Detect language for non-empty pages
            if cleaned_text.strip():
                try:
                    language = self._chunker.config  # just a guard; detect below
                    language = self._detect_language(cleaned_text)
                except Exception as exc:  # noqa: BLE001
                    language = "und"
                    logger.warning(
                        "ingestion.language filename=%s page=%d reason=%s",
                        filename,
                        orig_page.page_number,
                        exc,
                    )
            else:
                language = "und"

            page_obj = ChunkerPageText(
                page_number=orig_page.page_number,
                text=cleaned_text,
                filename=filename,
                pdf_id=pdf_id,
                language=language,
            )
            chunker_pages.append(page_obj)

        # ── Stage 5: Chunking (Req 5.3 — ≥1 chunk per non-empty page) ────────
        # Filter out fully empty pages before chunking; each non-empty page must
        # produce ≥1 chunk (guaranteed by the Chunker for non-zero-token input).
        non_empty_pages = [p for p in chunker_pages if p.text.strip()]

        if not non_empty_pages:
            # All cleaned pages ended up empty — already logged above as zero chars,
            # but guard here too.
            return IngestionResult(
                pdf_path=pdf_path,
                num_chunks=0,
                errors=run_errors,
                success=True,
            )

        try:
            chunks = self._chunker.chunk(non_empty_pages)
        except Exception as exc:  # noqa: BLE001
            entry = IngestionErrorEntry.now(
                filename=filename,
                stage="extraction",
                reason=f"Chunking failed: {exc}",
                page_number=None,
            )
            self._error_log.append(entry)
            run_errors.append(entry)
            logger.error(
                "ingestion.chunking filename=%s reason=%s",
                filename,
                exc,
            )
            return IngestionResult(
                pdf_path=pdf_path,
                num_chunks=0,
                errors=run_errors,
                success=True,
            )

        # ── Stage 6: Embed + upsert each chunk ───────────────────────────────
        for chunk in chunks:
            point_id = self._vector_store.compute_point_id(
                chunk.text, chunk.metadata
            )

            # Req 4.4: if identical chunk already exists, reuse vectors, skip embed
            try:
                already_exists = self._vector_store.exists(point_id)
            except Exception as exc:  # noqa: BLE001
                # Existence check failure: conservatively proceed to embed+upsert
                already_exists = False
                logger.warning(
                    "ingestion.exists_check filename=%s page=%d reason=%s",
                    filename,
                    chunk.metadata.page_number,
                    exc,
                )

            if already_exists:
                logger.debug(
                    "ingestion.skip_embed filename=%s page=%d chunk_pos=%d (already exists)",
                    filename,
                    chunk.metadata.page_number,
                    chunk.metadata.chunk_position,
                )
                num_chunks_upserted += 1  # count as successfully processed
                continue

            # Req 4.6: embed failure → log exactly one error, continue remaining chunks
            try:
                emb_results = self._embedding_model.embed([chunk.text])
                emb = emb_results[0]
            except Exception as exc:  # noqa: BLE001
                entry = IngestionErrorEntry.now(
                    filename=filename,
                    stage="embedding",
                    reason=str(exc),
                    page_number=chunk.metadata.page_number,
                )
                self._error_log.append(entry)
                run_errors.append(entry)
                logger.error(
                    "ingestion.embedding filename=%s page=%d reason=%s",
                    filename,
                    chunk.metadata.page_number,
                    exc,
                )
                continue  # skip this chunk, leave prior chunks unchanged (Req 4.7)

            # Req 4.7: persistence failure → log exactly one error, continue remaining chunks
            try:
                self._vector_store.upsert_chunk(chunk, emb)
                num_chunks_upserted += 1
            except Exception as exc:  # noqa: BLE001
                entry = IngestionErrorEntry.now(
                    filename=filename,
                    stage="persistence",
                    reason=str(exc),
                    page_number=chunk.metadata.page_number,
                )
                self._error_log.append(entry)
                run_errors.append(entry)
                logger.error(
                    "ingestion.persistence filename=%s page=%d reason=%s",
                    filename,
                    chunk.metadata.page_number,
                    exc,
                )
                # Prior chunks remain unchanged — we simply don't increment the counter
                # and continue with the next chunk (Req 4.7)

        return IngestionResult(
            pdf_path=pdf_path,
            num_chunks=num_chunks_upserted,
            errors=run_errors,
            success=True,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_language(self, text: str) -> str:
        """Delegate language detection to the chunker's language detector if
        available, or fall back to a module-level LanguageDetector instance.

        The language detector is not a direct constructor parameter here — the
        pipeline constructs a local instance on first use (cheap; stateless).
        """
        if not hasattr(self, "_language_detector"):
            try:
                from backend.ingestion.language import LanguageDetector  # noqa: PLC0415
            except ImportError:
                from ingestion.language import LanguageDetector  # type: ignore[no-redef]  # noqa: PLC0415
            self._language_detector = LanguageDetector()
        return self._language_detector.detect(text)

    @staticmethod
    def _derive_pdf_id(pdf_path: str) -> str:
        """Derive a stable, deterministic identifier for a PDF from its path.

        Uses a SHA-256 of the absolute-normalised path so that the same file
        always maps to the same id regardless of working directory, while
        different files (even with the same basename) map to different ids.

        This matches the idempotency requirement (Req 5.4): identical source
        files produce identical ``pdf_id`` values across runs.
        """
        normalised = os.path.normcase(os.path.abspath(pdf_path))
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()
