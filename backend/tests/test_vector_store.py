"""
backend/tests/test_vector_store.py

Unit tests for backend/vector_store.py

Covers:
  - compute_point_id: determinism, idempotency, uniqueness (Req 4.4, 5.4)
  - data_models: SparseVector, EmbeddingResult, ChunkMetadata, Chunk, Candidate

These tests do NOT require a live Qdrant server — they only exercise
the pure-logic portions of the VectorStore adapter.

Requirements: 4.3, 4.4, 4.5, 6.1
"""

from __future__ import annotations

import hashlib
import uuid

import pytest

from backend.data_models import (
    Candidate,
    Chunk,
    ChunkMetadata,
    EmbeddingResult,
    SparseVector,
)
from backend.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_metadata(
    pdf_id: str = "doc1",
    filename: str = "test.pdf",
    page_number: int = 1,
    chunk_position: int = 0,
    language: str = "en",
) -> ChunkMetadata:
    return ChunkMetadata(
        pdf_id=pdf_id,
        filename=filename,
        page_number=page_number,
        chunk_position=chunk_position,
        language=language,
    )


def make_chunk(text: str = "hello world", **meta_kwargs) -> Chunk:
    return Chunk(text=text, metadata=make_metadata(**meta_kwargs))


# ---------------------------------------------------------------------------
# compute_point_id tests
# ---------------------------------------------------------------------------

class TestComputePointId:
    """Property 15: Point ids are deterministic and idempotent (Req 4.4, 5.4)."""

    def test_returns_qdrant_compatible_uuid(self):
        """Point id is a valid UUID derived from the SHA-256 digest."""
        pid = VectorStore.compute_point_id("text", make_metadata())
        assert str(uuid.UUID(pid)) == pid

    def test_deterministic_same_inputs(self):
        """Same text + metadata always produces the same point id."""
        meta = make_metadata()
        pid1 = VectorStore.compute_point_id("hello world", meta)
        pid2 = VectorStore.compute_point_id("hello world", meta)
        assert pid1 == pid2

    def test_idempotent_repeated_calls(self):
        """Calling compute_point_id many times with identical inputs is idempotent."""
        meta = make_metadata(pdf_id="x", page_number=3, chunk_position=7)
        text = "the quick brown fox"
        ids = {VectorStore.compute_point_id(text, meta) for _ in range(20)}
        assert len(ids) == 1, "All calls should return the same id"

    def test_different_text_produces_different_id(self):
        """Different text → different point id."""
        meta = make_metadata()
        pid1 = VectorStore.compute_point_id("text A", meta)
        pid2 = VectorStore.compute_point_id("text B", meta)
        assert pid1 != pid2

    def test_different_pdf_id_produces_different_id(self):
        """Different pdf_id → different point id (same text and position)."""
        text = "same text"
        meta1 = make_metadata(pdf_id="doc1")
        meta2 = make_metadata(pdf_id="doc2")
        assert VectorStore.compute_point_id(text, meta1) != VectorStore.compute_point_id(text, meta2)

    def test_different_page_number_produces_different_id(self):
        """Different page_number → different point id."""
        text = "same text"
        meta1 = make_metadata(page_number=1)
        meta2 = make_metadata(page_number=2)
        assert VectorStore.compute_point_id(text, meta1) != VectorStore.compute_point_id(text, meta2)

    def test_different_chunk_position_produces_different_id(self):
        """Different chunk_position → different point id."""
        text = "same text"
        meta1 = make_metadata(chunk_position=0)
        meta2 = make_metadata(chunk_position=1)
        assert VectorStore.compute_point_id(text, meta1) != VectorStore.compute_point_id(text, meta2)

    def test_filename_does_not_affect_id(self):
        """filename is NOT part of the hash formula — same text + pdf_id + page + position
        should yield the same id regardless of filename (the formula uses pdf_id, not filename).
        """
        text = "same text"
        meta1 = make_metadata(filename="report_v1.pdf")
        meta2 = make_metadata(filename="report_v2.pdf")
        # Same pdf_id, page_number, chunk_position → same id even with different filenames
        assert VectorStore.compute_point_id(text, meta1) == VectorStore.compute_point_id(text, meta2)

    def test_matches_manual_sha256_uuid_prefix(self):
        """Point id is the UUID form of the first 128 SHA-256 bits."""
        text = "manual check"
        meta = make_metadata(pdf_id="pdfa", page_number=5, chunk_position=2)
        raw = "manual check\x00pdfa\x005\x002"
        sha256_hex = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        expected = str(uuid.UUID(hex=sha256_hex[:32]))
        assert VectorStore.compute_point_id(text, meta) == expected

    def test_identical_chunk_same_as_reingested(self):
        """Two Chunk objects with identical text + metadata produce the same id.

        This is the core idempotency guarantee: re-ingesting the same chunk
        must not create a duplicate point (Req 4.4).
        """
        chunk1 = make_chunk("overlap text", page_number=10, chunk_position=3)
        chunk2 = make_chunk("overlap text", page_number=10, chunk_position=3)
        pid1 = VectorStore.compute_point_id(chunk1.text, chunk1.metadata)
        pid2 = VectorStore.compute_point_id(chunk2.text, chunk2.metadata)
        assert pid1 == pid2

    def test_unicode_text_handled(self):
        """Unicode text (non-ASCII) produces a valid UUID without errors."""
        text = "日本語テキスト — 한국어 — Ελληνικά"
        meta = make_metadata(pdf_id="multilang")
        pid = VectorStore.compute_point_id(text, meta)
        assert str(uuid.UUID(pid)) == pid

    def test_empty_text_produces_valid_id(self):
        """Empty text still produces a valid UUID without error."""
        pid = VectorStore.compute_point_id("", make_metadata())
        assert str(uuid.UUID(pid)) == pid


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------

class TestSparseVector:
    def test_construction(self):
        sv = SparseVector(indices=[1, 42, 100], values=[0.5, 1.2, 0.3])
        assert sv.indices == [1, 42, 100]
        assert sv.values == [0.5, 1.2, 0.3]

    def test_frozen(self):
        sv = SparseVector(indices=[1], values=[0.5])
        with pytest.raises((AttributeError, TypeError)):
            sv.indices = [2]  # type: ignore[misc]

    def test_empty_sparse_vector(self):
        sv = SparseVector(indices=[], values=[])
        assert sv.indices == []
        assert sv.values == []


class TestEmbeddingResult:
    def test_construction(self):
        dense = [0.1] * 1024
        sparse = SparseVector(indices=[0, 5], values=[1.0, 0.5])
        er = EmbeddingResult(dense=dense, sparse=sparse)
        assert len(er.dense) == 1024
        assert er.sparse.indices == [0, 5]

    def test_frozen(self):
        er = EmbeddingResult(
            dense=[0.0] * 1024,
            sparse=SparseVector(indices=[], values=[]),
        )
        with pytest.raises((AttributeError, TypeError)):
            er.dense = []  # type: ignore[misc]


class TestChunkMetadata:
    def test_construction(self):
        meta = ChunkMetadata(
            pdf_id="abc123",
            filename="doc.pdf",
            page_number=3,
            chunk_position=1,
            language="fr",
        )
        assert meta.pdf_id == "abc123"
        assert meta.filename == "doc.pdf"
        assert meta.page_number == 3
        assert meta.chunk_position == 1
        assert meta.language == "fr"

    def test_frozen(self):
        meta = make_metadata()
        with pytest.raises((AttributeError, TypeError)):
            meta.page_number = 99  # type: ignore[misc]

    def test_equality(self):
        m1 = make_metadata(pdf_id="x", page_number=1)
        m2 = make_metadata(pdf_id="x", page_number=1)
        assert m1 == m2

    def test_inequality_different_page(self):
        m1 = make_metadata(page_number=1)
        m2 = make_metadata(page_number=2)
        assert m1 != m2


class TestChunk:
    def test_construction(self):
        chunk = Chunk(text="hello", metadata=make_metadata())
        assert chunk.text == "hello"

    def test_frozen(self):
        chunk = make_chunk()
        with pytest.raises((AttributeError, TypeError)):
            chunk.text = "changed"  # type: ignore[misc]


class TestCandidate:
    def test_construction(self):
        c = Candidate(text="passage", score=0.87, metadata=make_metadata())
        assert c.text == "passage"
        assert c.score == pytest.approx(0.87)

    def test_frozen(self):
        c = Candidate(text="t", score=1.0, metadata=make_metadata())
        with pytest.raises((AttributeError, TypeError)):
            c.score = 0.0  # type: ignore[misc]
