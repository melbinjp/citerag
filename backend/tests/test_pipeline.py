"""Unit tests for backend/ingestion/pipeline.py.

Tests cover:
- IngestionResult dataclass fields.
- ingest_pdf: corrupt/encrypted PDF → log error, return success=False, continue.
- ingest_pdf: all-zero-char pages → log error, return success=False.
- ingest_pdf: happy path → chunks upserted, success=True.
- ingest_pdf: idempotent skip when chunk already exists in VectorStore.
- ingest_pdf: embedding failure for one chunk → log error, continue remaining chunks.
- ingest_pdf: persistence failure for one chunk → log error, continue remaining chunks.
- ingest_pdf: exactly one error entry per failure (Req 13.4).
"""

import sys
import os
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, r"c:\Workspace_Melbin\fun_websites\ackathon\backend")

from ingestion.pipeline import IngestionPipeline, IngestionResult
from ingestion.error_log import IngestionErrorEntry, IngestionErrorLog
from ingestion.extract import ExtractionError


# ---------------------------------------------------------------------------
# Helpers: build minimal fake objects
# ---------------------------------------------------------------------------

def _make_page_text(page_number: int = 1, text: str = "Hello world") -> MagicMock:
    """Fake extract.PageText."""
    pt = MagicMock()
    pt.page_number = page_number
    pt.text = text
    return pt


def _make_chunk(page_number: int = 1, text: str = "chunk text", chunk_position: int = 0) -> MagicMock:
    """Fake data_models.Chunk with metadata."""
    meta = MagicMock()
    meta.page_number = page_number
    meta.chunk_position = chunk_position
    meta.language = "en"
    meta.filename = "test.pdf"
    meta.pdf_id = "abc123"

    chunk = MagicMock()
    chunk.text = text
    chunk.metadata = meta
    return chunk


def _make_embedding_result() -> MagicMock:
    emb = MagicMock()
    emb.dense = [0.1] * 1024
    emb.sparse = MagicMock()
    return emb


def _build_pipeline(
    extractor=None,
    cleaner=None,
    chunker=None,
    embedding_model=None,
    vector_store=None,
    error_log=None,
):
    """Build an IngestionPipeline with sensible mock defaults."""
    if extractor is None:
        extractor = MagicMock()
        extractor.extract.return_value = [_make_page_text()]

    if cleaner is None:
        cleaner = MagicMock()
        cleaner.clean_document.side_effect = lambda pages: pages  # identity

    if chunker is None:
        chunker = MagicMock()
        cfg = MagicMock()
        cfg.max_tokens = 800
        chunker.config = cfg
        chunker.chunk.return_value = [_make_chunk()]

    if embedding_model is None:
        embedding_model = MagicMock()
        embedding_model.embed.return_value = [_make_embedding_result()]

    if vector_store is None:
        vector_store = MagicMock()
        vector_store.exists.return_value = False
        vector_store.compute_point_id.return_value = "deadbeef" * 8
        vector_store.upsert_chunk.return_value = None

    if error_log is None:
        error_log = IngestionErrorLog()

    return IngestionPipeline(
        extractor=extractor,
        cleaner=cleaner,
        chunker=chunker,
        embedding_model=embedding_model,
        vector_store=vector_store,
        error_log=error_log,
    )


# ---------------------------------------------------------------------------
# IngestionResult
# ---------------------------------------------------------------------------

class TestIngestionResult:
    def test_fields(self):
        r = IngestionResult(pdf_path="a.pdf", num_chunks=3, errors=[], success=True)
        assert r.pdf_path == "a.pdf"
        assert r.num_chunks == 3
        assert r.errors == []
        assert r.success is True

    def test_success_false(self):
        r = IngestionResult(pdf_path="b.pdf", num_chunks=0, errors=[], success=False)
        assert r.success is False
        assert r.num_chunks == 0


# ---------------------------------------------------------------------------
# Extraction failure (Req 1.7): corrupt / encrypted PDF
# ---------------------------------------------------------------------------

class TestExtractionFailure:
    def test_extraction_error_returns_success_false(self):
        extractor = MagicMock()
        extractor.extract.side_effect = ExtractionError("file is encrypted")
        error_log = IngestionErrorLog()
        pipeline = _build_pipeline(extractor=extractor, error_log=error_log)

        result = pipeline.ingest_pdf("locked.pdf")

        assert result.success is False
        assert result.num_chunks == 0

    def test_extraction_error_logs_exactly_one_entry(self):
        extractor = MagicMock()
        extractor.extract.side_effect = ExtractionError("corrupt PDF")
        error_log = IngestionErrorLog()
        pipeline = _build_pipeline(extractor=extractor, error_log=error_log)

        pipeline.ingest_pdf("bad.pdf")

        assert len(error_log.entries) == 1
        assert error_log.entries[0].stage == "extraction"
        assert error_log.entries[0].filename == "bad.pdf"
        assert "corrupt PDF" in error_log.entries[0].reason

    def test_extraction_error_page_number_is_none(self):
        extractor = MagicMock()
        extractor.extract.side_effect = ExtractionError("oops")
        error_log = IngestionErrorLog()
        pipeline = _build_pipeline(extractor=extractor, error_log=error_log)

        pipeline.ingest_pdf("err.pdf")

        assert error_log.entries[0].page_number is None

    def test_extraction_error_entry_in_result(self):
        extractor = MagicMock()
        extractor.extract.side_effect = ExtractionError("bad")
        error_log = IngestionErrorLog()
        pipeline = _build_pipeline(extractor=extractor, error_log=error_log)

        result = pipeline.ingest_pdf("fail.pdf")

        assert len(result.errors) == 1
        assert result.errors[0].stage == "extraction"


# ---------------------------------------------------------------------------
# Zero-char document (Req 1.6, 5.5)
# ---------------------------------------------------------------------------

class TestZeroCharDocument:
    def test_all_empty_pages_logs_error_and_returns_false(self):
        extractor = MagicMock()
        extractor.extract.return_value = [
            _make_page_text(page_number=1, text=""),
            _make_page_text(page_number=2, text="   "),
        ]
        error_log = IngestionErrorLog()
        pipeline = _build_pipeline(extractor=extractor, error_log=error_log)

        result = pipeline.ingest_pdf("empty_doc.pdf")

        assert result.success is False
        assert result.num_chunks == 0
        assert len(error_log.entries) == 1
        assert error_log.entries[0].stage == "extraction"
        assert "zero" in error_log.entries[0].reason.lower()

    def test_zero_char_exactly_one_error_entry(self):
        extractor = MagicMock()
        extractor.extract.return_value = [_make_page_text(text="")]
        error_log = IngestionErrorLog()
        pipeline = _build_pipeline(extractor=extractor, error_log=error_log)

        pipeline.ingest_pdf("blank.pdf")

        assert len(error_log.entries) == 1


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_successful_ingestion_returns_correct_chunk_count(self):
        chunks = [_make_chunk(text="chunk A"), _make_chunk(text="chunk B", chunk_position=1)]
        chunker = MagicMock()
        cfg = MagicMock()
        cfg.max_tokens = 800
        chunker.config = cfg
        chunker.chunk.return_value = chunks

        embedding_model = MagicMock()
        embedding_model.embed.return_value = [_make_embedding_result()]

        vector_store = MagicMock()
        vector_store.exists.return_value = False
        vector_store.compute_point_id.return_value = "aabbcc" * 10
        vector_store.upsert_chunk.return_value = None

        pipeline = _build_pipeline(
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )
        result = pipeline.ingest_pdf("good.pdf")

        assert result.success is True
        assert result.num_chunks == 2
        assert result.errors == []

    def test_upsert_called_for_each_new_chunk(self):
        chunks = [_make_chunk(text="a"), _make_chunk(text="b", chunk_position=1)]
        chunker = MagicMock()
        cfg = MagicMock()
        cfg.max_tokens = 800
        chunker.config = cfg
        chunker.chunk.return_value = chunks

        embedding_model = MagicMock()
        embedding_model.embed.return_value = [_make_embedding_result()]

        vector_store = MagicMock()
        vector_store.exists.return_value = False
        vector_store.compute_point_id.return_value = "id"
        vector_store.upsert_chunk.return_value = None

        pipeline = _build_pipeline(
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )
        pipeline.ingest_pdf("doc.pdf")

        assert vector_store.upsert_chunk.call_count == 2

    def test_success_with_no_errors(self):
        pipeline = _build_pipeline()
        result = pipeline.ingest_pdf("clean.pdf")

        assert result.success is True
        assert result.errors == []


# ---------------------------------------------------------------------------
# Idempotent skip (Req 4.4)
# ---------------------------------------------------------------------------

class TestIdempotentSkip:
    def test_existing_chunk_skips_embedding(self):
        chunk = _make_chunk()
        chunker = MagicMock()
        cfg = MagicMock()
        cfg.max_tokens = 800
        chunker.config = cfg
        chunker.chunk.return_value = [chunk]

        embedding_model = MagicMock()
        vector_store = MagicMock()
        vector_store.exists.return_value = True          # chunk already in store
        vector_store.compute_point_id.return_value = "existingid"

        pipeline = _build_pipeline(
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )
        result = pipeline.ingest_pdf("dup.pdf")

        embedding_model.embed.assert_not_called()
        vector_store.upsert_chunk.assert_not_called()
        assert result.num_chunks == 1   # counted as processed
        assert result.errors == []

    def test_existing_chunk_not_counted_as_error(self):
        chunk = _make_chunk()
        chunker = MagicMock()
        cfg = MagicMock()
        cfg.max_tokens = 800
        chunker.config = cfg
        chunker.chunk.return_value = [chunk]

        vector_store = MagicMock()
        vector_store.exists.return_value = True
        vector_store.compute_point_id.return_value = "existing"

        error_log = IngestionErrorLog()
        pipeline = _build_pipeline(
            chunker=chunker,
            vector_store=vector_store,
            error_log=error_log,
        )
        pipeline.ingest_pdf("dup.pdf")

        assert len(error_log.entries) == 0

    def test_mix_new_and_existing_chunks(self):
        chunk_new = _make_chunk(text="new text", chunk_position=0)
        chunk_existing = _make_chunk(text="old text", chunk_position=1)
        chunker = MagicMock()
        cfg = MagicMock()
        cfg.max_tokens = 800
        chunker.config = cfg
        chunker.chunk.return_value = [chunk_new, chunk_existing]

        embedding_model = MagicMock()
        embedding_model.embed.return_value = [_make_embedding_result()]

        call_count = [0]

        def _exists(point_id):
            call_count[0] += 1
            # First chunk (new): not exists; second chunk (existing): exists
            return call_count[0] > 1

        vector_store = MagicMock()
        vector_store.exists.side_effect = _exists
        vector_store.compute_point_id.return_value = "id"
        vector_store.upsert_chunk.return_value = None

        pipeline = _build_pipeline(
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )
        result = pipeline.ingest_pdf("mixed.pdf")

        assert embedding_model.embed.call_count == 1   # only called for the new chunk
        assert vector_store.upsert_chunk.call_count == 1
        assert result.num_chunks == 2   # both count as processed
        assert result.errors == []


# ---------------------------------------------------------------------------
# Embedding failure (Req 4.6, 13.4)
# ---------------------------------------------------------------------------

class TestEmbeddingFailure:
    def test_embedding_failure_logs_exactly_one_error(self):
        chunk = _make_chunk()
        chunker = MagicMock()
        cfg = MagicMock()
        cfg.max_tokens = 800
        chunker.config = cfg
        chunker.chunk.return_value = [chunk]

        embedding_model = MagicMock()
        embedding_model.embed.side_effect = RuntimeError("GPU OOM")

        error_log = IngestionErrorLog()
        pipeline = _build_pipeline(
            chunker=chunker,
            embedding_model=embedding_model,
            error_log=error_log,
        )
        pipeline.ingest_pdf("doc.pdf")

        assert len(error_log.entries) == 1
        assert error_log.entries[0].stage == "embedding"
        assert "GPU OOM" in error_log.entries[0].reason

    def test_embedding_failure_logs_correct_page_number(self):
        chunk = _make_chunk(page_number=5)
        chunker = MagicMock()
        cfg = MagicMock()
        cfg.max_tokens = 800
        chunker.config = cfg
        chunker.chunk.return_value = [chunk]

        embedding_model = MagicMock()
        embedding_model.embed.side_effect = RuntimeError("OOM")

        error_log = IngestionErrorLog()
        pipeline = _build_pipeline(
            chunker=chunker,
            embedding_model=embedding_model,
            error_log=error_log,
        )
        pipeline.ingest_pdf("doc.pdf")

        assert error_log.entries[0].page_number == 5

    def test_embedding_failure_continues_remaining_chunks(self):
        chunk_bad = _make_chunk(text="bad", chunk_position=0)
        chunk_good = _make_chunk(text="good", chunk_position=1)
        chunker = MagicMock()
        cfg = MagicMock()
        cfg.max_tokens = 800
        chunker.config = cfg
        chunker.chunk.return_value = [chunk_bad, chunk_good]

        call_count = [0]

        def _embed(texts):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("embed fail")
            return [_make_embedding_result()]

        embedding_model = MagicMock()
        embedding_model.embed.side_effect = _embed

        vector_store = MagicMock()
        vector_store.exists.return_value = False
        vector_store.compute_point_id.return_value = "id"
        vector_store.upsert_chunk.return_value = None

        error_log = IngestionErrorLog()
        pipeline = _build_pipeline(
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
            error_log=error_log,
        )
        result = pipeline.ingest_pdf("doc.pdf")

        # Second chunk should still be upserted
        assert vector_store.upsert_chunk.call_count == 1
        assert result.num_chunks == 1
        assert len(error_log.entries) == 1  # exactly one error

    def test_multiple_embedding_failures_log_one_error_each(self):
        chunks = [_make_chunk(text=f"chunk {i}", chunk_position=i) for i in range(3)]
        chunker = MagicMock()
        cfg = MagicMock()
        cfg.max_tokens = 800
        chunker.config = cfg
        chunker.chunk.return_value = chunks

        embedding_model = MagicMock()
        embedding_model.embed.side_effect = RuntimeError("always fail")

        error_log = IngestionErrorLog()
        pipeline = _build_pipeline(
            chunker=chunker,
            embedding_model=embedding_model,
            error_log=error_log,
        )
        pipeline.ingest_pdf("doc.pdf")

        # Exactly one error per chunk failure
        assert len(error_log.entries) == 3


# ---------------------------------------------------------------------------
# Persistence failure (Req 4.7, 13.4)
# ---------------------------------------------------------------------------

class TestPersistenceFailure:
    def test_persistence_failure_logs_exactly_one_error(self):
        chunk = _make_chunk()
        chunker = MagicMock()
        cfg = MagicMock()
        cfg.max_tokens = 800
        chunker.config = cfg
        chunker.chunk.return_value = [chunk]

        embedding_model = MagicMock()
        embedding_model.embed.return_value = [_make_embedding_result()]

        vector_store = MagicMock()
        vector_store.exists.return_value = False
        vector_store.compute_point_id.return_value = "id"
        vector_store.upsert_chunk.side_effect = RuntimeError("Qdrant down")

        error_log = IngestionErrorLog()
        pipeline = _build_pipeline(
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
            error_log=error_log,
        )
        pipeline.ingest_pdf("doc.pdf")

        assert len(error_log.entries) == 1
        assert error_log.entries[0].stage == "persistence"
        assert "Qdrant down" in error_log.entries[0].reason

    def test_persistence_failure_leaves_prior_chunks_unchanged(self):
        """Prior successful upserts must NOT be rolled back on a later failure."""
        chunk_ok = _make_chunk(text="ok", chunk_position=0)
        chunk_fail = _make_chunk(text="fail", chunk_position=1)
        chunker = MagicMock()
        cfg = MagicMock()
        cfg.max_tokens = 800
        chunker.config = cfg
        chunker.chunk.return_value = [chunk_ok, chunk_fail]

        embedding_model = MagicMock()
        embedding_model.embed.return_value = [_make_embedding_result()]

        call_count = [0]

        def _upsert(chunk, emb):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("second upsert fails")

        vector_store = MagicMock()
        vector_store.exists.return_value = False
        vector_store.compute_point_id.return_value = "id"
        vector_store.upsert_chunk.side_effect = _upsert

        error_log = IngestionErrorLog()
        pipeline = _build_pipeline(
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
            error_log=error_log,
        )
        result = pipeline.ingest_pdf("doc.pdf")

        # First upsert succeeded; second failed
        assert vector_store.upsert_chunk.call_count == 2
        assert result.num_chunks == 1   # only the first was confirmed
        assert len(error_log.entries) == 1
        assert error_log.entries[0].stage == "persistence"

    def test_persistence_failure_continues_next_chunk(self):
        chunks = [_make_chunk(text=f"chunk {i}", chunk_position=i) for i in range(3)]
        chunker = MagicMock()
        cfg = MagicMock()
        cfg.max_tokens = 800
        chunker.config = cfg
        chunker.chunk.return_value = chunks

        embedding_model = MagicMock()
        embedding_model.embed.return_value = [_make_embedding_result()]

        call_count = [0]

        def _upsert(chunk, emb):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("middle upsert fails")

        vector_store = MagicMock()
        vector_store.exists.return_value = False
        vector_store.compute_point_id.return_value = "id"
        vector_store.upsert_chunk.side_effect = _upsert

        error_log = IngestionErrorLog()
        pipeline = _build_pipeline(
            chunker=chunker,
            embedding_model=embedding_model,
            vector_store=vector_store,
            error_log=error_log,
        )
        result = pipeline.ingest_pdf("doc.pdf")

        # All three chunks attempted; first and third succeeded
        assert vector_store.upsert_chunk.call_count == 3
        assert result.num_chunks == 2
        assert len(error_log.entries) == 1


# ---------------------------------------------------------------------------
# Error entry details (Req 13.4)
# ---------------------------------------------------------------------------

class TestErrorEntryDetails:
    def test_error_entry_has_filename(self):
        extractor = MagicMock()
        extractor.extract.side_effect = ExtractionError("oops")
        error_log = IngestionErrorLog()
        pipeline = _build_pipeline(extractor=extractor, error_log=error_log)

        pipeline.ingest_pdf("/some/path/my_report.pdf")

        assert error_log.entries[0].filename == "my_report.pdf"

    def test_error_entry_has_timestamp(self):
        extractor = MagicMock()
        extractor.extract.side_effect = ExtractionError("oops")
        error_log = IngestionErrorLog()
        pipeline = _build_pipeline(extractor=extractor, error_log=error_log)

        pipeline.ingest_pdf("doc.pdf")

        assert error_log.entries[0].timestamp  # non-empty ISO 8601 string
        assert "T" in error_log.entries[0].timestamp

    def test_result_errors_match_log_entries(self):
        """result.errors must contain the same entries that were appended to the log."""
        extractor = MagicMock()
        extractor.extract.side_effect = ExtractionError("bad")
        error_log = IngestionErrorLog()
        pipeline = _build_pipeline(extractor=extractor, error_log=error_log)

        result = pipeline.ingest_pdf("doc.pdf")

        assert result.errors == error_log.entries
