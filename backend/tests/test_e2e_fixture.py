"""
backend/tests/test_e2e_fixture.py

End-to-end integration test: fixture corpus ingestion → retrieve → generate → cite

This test wires the full RAG pipeline together over a small in-memory fixture
corpus using mocked external services (no live Qdrant, no live LLM, no model
download required).

Pipeline path tested:
    IngestionPipeline → RetrievalService → AnswerGenerator (build_citations)

Requirements validated:
  5.7 — Retrieval_Service returns results without requiring user upload
         (uses the pre-populated mock VectorStore)
  6.2 — Retrieval_Service embeds the query and performs hybrid search
  8.1 — AnswerGenerator uses a single-pass CoT prompt (one provider call)
  9.1 — System returns a complete answer with citations (functional assertion;
         latency is not measured here — that is covered by perf tests)

The mock VectorStore.hybrid_search() returns predefined Candidate objects that
simulate real retrieval results, so the test validates wiring rather than
vector similarity.  The mock EmbeddingModel returns plausible-shaped vectors.
The stub LLMProvider returns a canned answer token.

Usage::

    pytest backend/tests/test_e2e_fixture.py -v
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Module-level imports that rely on conftest.py having already patched sys.path
# ---------------------------------------------------------------------------
from backend.data_models import (
    Candidate,
    ChunkMetadata,
    EmbeddingResult,
    SparseVector,
)
from backend.generation import AnswerGenerator
from backend.ingestion.error_log import IngestionErrorLog
from backend.ingestion.pipeline import IngestionPipeline
from backend.retrieval import RetrievalService


# ===========================================================================
# Helpers / builders
# ===========================================================================

def _make_sparse(indices: list[int] | None = None, values: list[float] | None = None) -> SparseVector:
    """Return a SparseVector with default values if not supplied."""
    return SparseVector(
        indices=indices or [1, 42, 200],
        values=values or [0.8, 0.5, 0.2],
    )


def _make_embedding_result(dim: int = 1024) -> EmbeddingResult:
    """Return a plausible-shaped EmbeddingResult (dense 1024-d + sparse)."""
    return EmbeddingResult(
        dense=[0.1] * dim,
        sparse=_make_sparse(),
    )


def _make_chunk_metadata(
    pdf_id: str = "fixture_pdf_001",
    filename: str = "fixture.pdf",
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


def _make_candidate(
    text: str,
    score: float = 0.75,
    filename: str = "fixture.pdf",
    page_number: int = 1,
) -> Candidate:
    return Candidate(
        text=text,
        score=score,
        metadata=_make_chunk_metadata(filename=filename, page_number=page_number),
    )


# ---------------------------------------------------------------------------
# Fixture corpus: small set of simulated page texts
# ---------------------------------------------------------------------------

FIXTURE_PAGES = [
    {
        "page_number": 1,
        "text": (
            "The photosynthesis process converts sunlight into chemical energy. "
            "Chlorophyll absorbs light primarily in the blue and red wavelengths."
        ),
    },
    {
        "page_number": 2,
        "text": (
            "Plants release oxygen as a by-product of photosynthesis. "
            "The Calvin cycle produces glucose from carbon dioxide."
        ),
    },
    {
        "page_number": 3,
        "text": (
            "Mitochondria are the powerhouses of the cell. "
            "They convert glucose into ATP through cellular respiration."
        ),
    },
]

FIXTURE_FILENAME = "biology_notes.pdf"

# Predefined search results that the mock VectorStore returns for any query —
# simulates what real hybrid search would return from the ingested corpus.
SIMULATED_SEARCH_RESULTS = [
    _make_candidate(
        text=FIXTURE_PAGES[0]["text"],
        score=0.92,
        filename=FIXTURE_FILENAME,
        page_number=1,
    ),
    _make_candidate(
        text=FIXTURE_PAGES[1]["text"],
        score=0.81,
        filename=FIXTURE_FILENAME,
        page_number=2,
    ),
]


# ===========================================================================
# Stub LLMProvider
# ===========================================================================

class _StubLLMProvider:
    """Minimal LLMProvider stub that returns a canned single-token answer.

    Satisfies the LLMProvider protocol expected by AnswerGenerator.
    """

    name = "stub"

    def __init__(self, answer: str = "Photosynthesis converts sunlight into energy.") -> None:
        self._answer = answer
        self.call_count: int = 0
        self.last_prompt: str | None = None

    async def generate(
        self,
        prompt: str,
        *,
        timeout_s: float,
        stream: bool,
    ) -> AsyncIterator[str]:
        self.call_count += 1
        self.last_prompt = prompt
        return self._single_token_gen()

    async def _single_token_gen(self) -> AsyncIterator[str]:
        yield self._answer


# ===========================================================================
# Mock VectorStore and EmbeddingModel builders
# ===========================================================================

def _build_mock_vector_store(
    predefined_candidates: list[Candidate] | None = None,
) -> MagicMock:
    """Return a mock VectorStore that:
    - exists() returns False (so every chunk is treated as new)
    - compute_point_id() returns a deterministic fake id
    - upsert_chunk() is a no-op (records the call)
    - hybrid_search() returns the predefined candidates
    """
    store = MagicMock()
    store.exists.return_value = False
    store.compute_point_id.side_effect = lambda text, metadata: (
        f"{metadata.filename}:{metadata.page_number}:{metadata.chunk_position}"
    )
    store.upsert_chunk.return_value = None
    store.hybrid_search.return_value = (
        predefined_candidates if predefined_candidates is not None
        else SIMULATED_SEARCH_RESULTS
    )
    return store


def _build_mock_embedding_model() -> MagicMock:
    """Return a mock EmbeddingModel whose embed() returns one result per input."""
    model = MagicMock()
    model.embed.side_effect = lambda texts: [_make_embedding_result() for _ in texts]
    return model


# ===========================================================================
# Ingestion-side helpers
# ===========================================================================

def _build_mock_extractor(pages: list[dict] | None = None, filename: str = FIXTURE_FILENAME) -> MagicMock:
    """Return a mock Text_Extractor that yields fake PageText objects."""
    if pages is None:
        pages = FIXTURE_PAGES

    page_texts = []
    for p in pages:
        pt = MagicMock()
        pt.page_number = p["page_number"]
        pt.text = p["text"]
        page_texts.append(pt)

    extractor = MagicMock()
    extractor.extract.return_value = page_texts
    return extractor


def _build_mock_cleaner() -> MagicMock:
    """Return a mock Cleaner that is an identity function (pass-through)."""
    cleaner = MagicMock()
    cleaner.clean_document.side_effect = lambda texts: texts  # identity
    return cleaner


def _build_mock_chunker(pages: list[dict] | None = None, filename: str = FIXTURE_FILENAME) -> MagicMock:
    """Return a mock Chunker that creates one Chunk per page."""
    if pages is None:
        pages = FIXTURE_PAGES

    from backend.data_models import Chunk

    chunks = []
    pdf_id = "fixture_e2e_pdf_id"
    for i, p in enumerate(pages):
        meta = ChunkMetadata(
            pdf_id=pdf_id,
            filename=filename,
            page_number=p["page_number"],
            chunk_position=i,
            language="en",
        )
        chunks.append(Chunk(text=p["text"], metadata=meta))

    chunker = MagicMock()
    cfg = MagicMock()
    cfg.max_tokens = 800
    chunker.config = cfg
    chunker.chunk.return_value = chunks
    return chunker


# ===========================================================================
# Test: Full end-to-end fixture pipeline
# ===========================================================================

class TestE2EFixturePipeline:
    """End-to-end wiring test: ingest → retrieve → generate → cite.

    All external I/O (Qdrant, embedding model, LLM) is mocked.  The test
    validates that the pipeline components are correctly connected and that
    the output satisfies the structural requirements (Req 5.7, 6.2, 8.1, 9.1).
    """

    # ------------------------------------------------------------------
    # Setup: build and run ingestion
    # ------------------------------------------------------------------

    def _build_pipeline_and_services(
        self,
        extra_candidates: list[Candidate] | None = None,
    ) -> tuple[IngestionPipeline, RetrievalService, AnswerGenerator, MagicMock, MagicMock, _StubLLMProvider]:
        """Construct all pipeline components backed by mocks."""
        mock_store = _build_mock_vector_store(extra_candidates)
        mock_emb_model = _build_mock_embedding_model()

        extractor = _build_mock_extractor()
        cleaner = _build_mock_cleaner()
        chunker = _build_mock_chunker()
        error_log = IngestionErrorLog()

        pipeline = IngestionPipeline(
            extractor=extractor,
            cleaner=cleaner,
            chunker=chunker,
            embedding_model=mock_emb_model,
            vector_store=mock_store,
            error_log=error_log,
        )

        retrieval_service = RetrievalService(
            embedding_model=mock_emb_model,
            vector_store=mock_store,
        )

        stub_provider = _StubLLMProvider()
        answer_generator = AnswerGenerator(provider=stub_provider)

        return pipeline, retrieval_service, answer_generator, mock_store, mock_emb_model, stub_provider

    # ------------------------------------------------------------------
    # 1. Ingestion phase tests (Req 5.7: corpus persisted without re-upload)
    # ------------------------------------------------------------------

    def test_ingestion_succeeds_and_upserts_all_pages(self):
        """Ingesting the fixture corpus must succeed and upsert one chunk per page."""
        pipeline, _, _, mock_store, _, _ = self._build_pipeline_and_services()

        result = pipeline.ingest_pdf(f"corpus/{FIXTURE_FILENAME}")

        assert result.success is True, f"Ingestion failed: {result.errors}"
        assert result.num_chunks == len(FIXTURE_PAGES)
        assert result.errors == []

    def test_ingestion_calls_upsert_for_each_chunk(self):
        """Each page chunk must be upserted into the VectorStore."""
        pipeline, _, _, mock_store, _, _ = self._build_pipeline_and_services()

        pipeline.ingest_pdf(f"corpus/{FIXTURE_FILENAME}")

        assert mock_store.upsert_chunk.call_count == len(FIXTURE_PAGES)

    def test_ingestion_embeds_each_chunk(self):
        """The embedding model must be called for each new chunk."""
        pipeline, _, _, _, mock_emb_model, _ = self._build_pipeline_and_services()

        pipeline.ingest_pdf(f"corpus/{FIXTURE_FILENAME}")

        # embed is called once per chunk (3 pages → 3 embed calls)
        assert mock_emb_model.embed.call_count == len(FIXTURE_PAGES)

    # ------------------------------------------------------------------
    # 2. Retrieval phase tests (Req 5.7, 6.2)
    # ------------------------------------------------------------------

    def test_retrieval_returns_candidates_without_upload(self):
        """Req 5.7: retrieve from the pre-populated store without uploading documents."""
        _, retrieval_service, _, mock_store, _, _ = self._build_pipeline_and_services()

        result = retrieval_service.retrieve("What is photosynthesis?", top_k=5)

        # hybrid_search is called — retrieval uses the persisted store
        mock_store.hybrid_search.assert_called_once()
        assert len(result.candidates) > 0
        assert result.error is None

    def test_retrieval_embeds_query(self):
        """Req 6.2: the query is embedded before search."""
        _, retrieval_service, _, _, mock_emb_model, _ = self._build_pipeline_and_services()
        # Reset call count from ingestion
        mock_emb_model.embed.reset_mock()

        retrieval_service.retrieve("What is photosynthesis?")

        mock_emb_model.embed.assert_called_once_with(["What is photosynthesis?"])

    def test_retrieval_candidates_have_text_score_and_metadata(self):
        """Req 6.4: every candidate must carry text, score, and metadata."""
        _, retrieval_service, _, _, _, _ = self._build_pipeline_and_services()

        result = retrieval_service.retrieve("Tell me about plants.")

        for candidate in result.candidates:
            assert isinstance(candidate.text, str)
            assert len(candidate.text) > 0
            assert isinstance(candidate.score, float)
            assert candidate.metadata is not None
            assert candidate.metadata.filename
            assert candidate.metadata.page_number >= 1

    def test_retrieval_does_not_write_to_store(self):
        """Retrieval must be read-only — no upsert during query answering."""
        pipeline, retrieval_service, _, mock_store, _, _ = self._build_pipeline_and_services()
        # First ingest to populate
        pipeline.ingest_pdf(f"corpus/{FIXTURE_FILENAME}")
        upsert_calls_after_ingest = mock_store.upsert_chunk.call_count

        # Now retrieve
        retrieval_service.retrieve("What do chloroplasts do?")

        assert mock_store.upsert_chunk.call_count == upsert_calls_after_ingest, (
            "Retrieval must not write to the vector store"
        )

    # ------------------------------------------------------------------
    # 3. Generation phase tests (Req 8.1)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_generate_returns_non_empty_answer(self):
        """Req 8.1: generation must return a non-empty answer for non-empty chunks."""
        _, retrieval_service, answer_generator, _, _, _ = self._build_pipeline_and_services()

        retrieval_result = retrieval_service.retrieve("What is photosynthesis?")
        generation_result = await answer_generator.generate(
            query="What is photosynthesis?",
            chunks=retrieval_result.candidates,
            history=[],
        )

        assert generation_result.error is None, f"Generation error: {generation_result.error}"
        assert generation_result.answer, "Answer must not be empty"
        assert len(generation_result.answer) > 0

    @pytest.mark.asyncio
    async def test_generate_invokes_provider_exactly_once(self):
        """Req 8.1/8.2: single-pass CoT — provider must be called exactly once."""
        _, retrieval_service, answer_generator, _, _, stub_provider = (
            self._build_pipeline_and_services()
        )

        retrieval_result = retrieval_service.retrieve("What do plants release?")
        await answer_generator.generate(
            query="What do plants release?",
            chunks=retrieval_result.candidates,
            history=[],
        )

        assert stub_provider.call_count == 1, (
            "AnswerGenerator must call the LLM provider exactly once (Req 8.2)"
        )

    @pytest.mark.asyncio
    async def test_generate_prompt_contains_retrieved_chunks(self):
        """Req 8.1: the CoT prompt must inject the retrieved chunks."""
        _, retrieval_service, answer_generator, _, _, stub_provider = (
            self._build_pipeline_and_services()
        )

        retrieval_result = retrieval_service.retrieve("What is ATP?")
        await answer_generator.generate(
            query="What is ATP?",
            chunks=retrieval_result.candidates,
            history=[],
        )

        assert stub_provider.last_prompt is not None
        # At least one chunk's text must appear in the prompt
        found_chunk = any(
            candidate.text in stub_provider.last_prompt
            for candidate in retrieval_result.candidates
        )
        assert found_chunk, "CoT prompt must contain the retrieved chunk texts"

    # ------------------------------------------------------------------
    # 4. Citations phase tests (Req 8.4, 9.1)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_citations_are_present_after_generation(self):
        """Req 9.1 / 8.4: complete answer must include structured citations."""
        _, retrieval_service, answer_generator, _, _, _ = self._build_pipeline_and_services()

        retrieval_result = retrieval_service.retrieve("What is photosynthesis?")
        generation_result = await answer_generator.generate(
            query="What is photosynthesis?",
            chunks=retrieval_result.candidates,
            history=[],
        )

        assert generation_result.citations, "Citations must not be empty"
        for citation in generation_result.citations:
            assert citation.filename, "Each citation must have a filename"
            assert len(citation.pages) > 0, "Each citation must reference at least one page"

    @pytest.mark.asyncio
    async def test_citations_reference_fixture_filename(self):
        """Citations must point to the fixture PDF filename."""
        _, retrieval_service, answer_generator, _, _, _ = self._build_pipeline_and_services()

        retrieval_result = retrieval_service.retrieve("Tell me about chlorophyll.")
        generation_result = await answer_generator.generate(
            query="Tell me about chlorophyll.",
            chunks=retrieval_result.candidates,
            history=[],
        )

        cited_filenames = {c.filename for c in generation_result.citations}
        assert FIXTURE_FILENAME in cited_filenames, (
            f"Expected citation for '{FIXTURE_FILENAME}', got {cited_filenames}"
        )

    @pytest.mark.asyncio
    async def test_citations_pages_are_sorted_and_unique(self):
        """Citations must have sorted, deduplicated page numbers."""
        _, retrieval_service, answer_generator, _, _, _ = self._build_pipeline_and_services()

        retrieval_result = retrieval_service.retrieve("Describe cell biology.")
        generation_result = await answer_generator.generate(
            query="Describe cell biology.",
            chunks=retrieval_result.candidates,
            history=[],
        )

        for citation in generation_result.citations:
            pages = citation.pages
            assert pages == sorted(set(pages)), (
                f"Citation pages must be sorted and unique; got {pages}"
            )

    def test_build_citations_directly_from_candidates(self):
        """build_citations must derive one Citation per unique filename."""
        _, _, answer_generator, _, _, _ = self._build_pipeline_and_services()

        citations = answer_generator.build_citations(SIMULATED_SEARCH_RESULTS)

        assert len(citations) == 1, (
            "Both simulated candidates share the same filename, expect one citation"
        )
        assert citations[0].filename == FIXTURE_FILENAME
        assert sorted(citations[0].pages) == citations[0].pages  # sorted

    # ------------------------------------------------------------------
    # 5. Full pipeline round-trip (ingest + retrieve + generate + cite)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_full_pipeline_round_trip(self):
        """The complete ingest → retrieve → generate → cite path produces a valid result.

        This is the primary end-to-end assertion:
        - Ingest a 3-page fixture corpus
        - Retrieve top-5 candidates for a sample query
        - Generate an answer using a stub LLM
        - Confirm: answer is non-empty, citations present, no errors

        Requirements: 5.7, 6.2, 8.1, 9.1
        """
        pipeline, retrieval_service, answer_generator, mock_store, _, stub_provider = (
            self._build_pipeline_and_services()
        )

        # --- Step 1: Ingest fixture corpus (Req 5.7) ---
        ingest_result = pipeline.ingest_pdf(f"corpus/{FIXTURE_FILENAME}")
        assert ingest_result.success is True, f"Ingestion failed: {ingest_result.errors}"
        assert ingest_result.num_chunks > 0

        # --- Step 2: Retrieve without re-upload (Req 5.7, 6.2) ---
        query = "How do plants produce energy from sunlight?"
        retrieval_result = retrieval_service.retrieve(query, top_k=5)

        assert retrieval_result.error is None
        assert len(retrieval_result.candidates) > 0, (
            "Retrieval must return candidates from the pre-ingested corpus (Req 5.7)"
        )

        # Confirm embedding was called for the query (Req 6.2)
        # (mock_emb_model.embed was also called during ingestion; check hybrid_search was called)
        mock_store.hybrid_search.assert_called()

        # --- Step 3: Generate single-pass CoT answer (Req 8.1) ---
        generation_result = await answer_generator.generate(
            query=query,
            chunks=retrieval_result.candidates,
            history=[],
        )

        assert generation_result.error is None, f"Generation error: {generation_result.error}"
        assert generation_result.answer, "Answer must be non-empty (Req 8.1)"

        # Provider called exactly once (Req 8.2 — no agentic multi-step)
        assert stub_provider.call_count == 1

        # --- Step 4: Verify citations (Req 9.1) ---
        assert generation_result.citations, "Citations must be present (Req 9.1)"
        for citation in generation_result.citations:
            assert citation.filename
            assert citation.pages

        # --- Summary: all assertions passed ---
        # No errors, non-empty answer, citations present — end-to-end wiring is correct

    # ------------------------------------------------------------------
    # 6. Edge case: empty store → no candidates → "no relevant info" response
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_empty_store_returns_no_info_response(self):
        """When the VectorStore has no chunks, generation returns the no-info response."""
        from backend.generation import _NO_INFO_ANSWER

        # Build with empty search results
        pipeline, retrieval_service, answer_generator, _, _, _ = (
            self._build_pipeline_and_services(extra_candidates=[])
        )

        retrieval_result = retrieval_service.retrieve("Some query")
        assert retrieval_result.candidates == []

        generation_result = await answer_generator.generate(
            query="Some query",
            chunks=retrieval_result.candidates,
            history=[],
        )

        assert generation_result.answer == _NO_INFO_ANSWER
        assert generation_result.citations == []
        assert generation_result.error is None

    # ------------------------------------------------------------------
    # 7. Idempotency: re-ingesting the same corpus does not double-upsert
    # ------------------------------------------------------------------

    def test_reingestion_is_idempotent(self):
        """Req 4.4 / 5.4: ingesting the same corpus twice skips existing chunks."""
        pipeline, _, _, mock_store, mock_emb_model, _ = self._build_pipeline_and_services()

        # First ingestion
        pipeline.ingest_pdf(f"corpus/{FIXTURE_FILENAME}")
        first_upsert_count = mock_store.upsert_chunk.call_count

        # Simulate second ingestion: store now reports all chunks as existing
        mock_store.exists.return_value = True
        mock_emb_model.embed.reset_mock()
        mock_store.upsert_chunk.reset_mock()

        result2 = pipeline.ingest_pdf(f"corpus/{FIXTURE_FILENAME}")

        # No new embeddings or upserts — all chunks already existed
        mock_emb_model.embed.assert_not_called()
        mock_store.upsert_chunk.assert_not_called()
        assert result2.success is True
        assert result2.errors == []
