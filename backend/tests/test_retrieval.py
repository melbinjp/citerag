"""
backend/tests/test_retrieval.py

Unit tests for backend/retrieval.py

Covers:
  - InvalidQueryError: raised before embedding for empty/too-long queries
    and out-of-range top_k (Req 6.6)
  - Query validation: boundary values for query length and top_k
  - Happy path: valid query → embedding called once → hybrid_search called once
    → RetrievalResult with candidates in hybrid-search order (Req 6.2, 6.3, 6.4)
  - Empty result when Vector_Store returns no candidates (Req 6.5)
  - Persistence requirement: service uses injected store without re-upload (Req 5.7)

All external dependencies (EmbeddingModel, VectorStore) are replaced with
lightweight mock objects — no Qdrant server or model download is needed.

Requirements: 5.7, 6.2, 6.3, 6.4, 6.5, 6.6
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from backend.data_models import Candidate, ChunkMetadata, EmbeddingResult, SparseVector
from backend.retrieval import InvalidQueryError, RetrievalResult, RetrievalService


# ---------------------------------------------------------------------------
# Helpers / fixtures
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


def make_candidate(text: str = "chunk text", score: float = 0.5) -> Candidate:
    return Candidate(text=text, score=score, metadata=make_metadata())


def make_embedding_result(
    dense_dim: int = 1024,
    sparse_indices: list[int] | None = None,
    sparse_values: list[float] | None = None,
) -> EmbeddingResult:
    if sparse_indices is None:
        sparse_indices = [0, 5, 42]
    if sparse_values is None:
        sparse_values = [0.9, 0.4, 0.1]
    return EmbeddingResult(
        dense=[0.1] * dense_dim,
        sparse=SparseVector(indices=sparse_indices, values=sparse_values),
    )


def make_mock_embedding_model(
    result: EmbeddingResult | None = None,
) -> MagicMock:
    """Return a mock EmbeddingModel whose embed() returns [result]."""
    mock = MagicMock()
    emb = result if result is not None else make_embedding_result()
    mock.embed.return_value = [emb]
    return mock


def make_mock_vector_store(
    candidates: list[Candidate] | None = None,
) -> MagicMock:
    """Return a mock VectorStore whose hybrid_search() returns candidates."""
    mock = MagicMock()
    mock.hybrid_search.return_value = candidates if candidates is not None else []
    return mock


def make_service(
    candidates: list[Candidate] | None = None,
    emb_result: EmbeddingResult | None = None,
) -> tuple[RetrievalService, MagicMock, MagicMock]:
    """Convenience factory returning (service, mock_emb, mock_store)."""
    mock_emb = make_mock_embedding_model(emb_result)
    mock_store = make_mock_vector_store(candidates)
    service = RetrievalService(
        embedding_model=mock_emb,
        vector_store=mock_store,
    )
    return service, mock_emb, mock_store


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_max_query_chars(self):
        assert RetrievalService.MAX_QUERY_CHARS == 4000

    def test_default_top_k(self):
        assert RetrievalService.DEFAULT_TOP_K == 5

    def test_max_top_k(self):
        assert RetrievalService.MAX_TOP_K == 100


# ---------------------------------------------------------------------------
# Query validation — empty query (Req 6.6)
# ---------------------------------------------------------------------------

class TestInvalidQueryEmpty:
    """Empty query must raise InvalidQueryError before embedding (Req 6.6)."""

    def test_empty_string_raises(self):
        service, mock_emb, _ = make_service()
        with pytest.raises(InvalidQueryError):
            service.retrieve("")

    def test_empty_query_does_not_call_embed(self):
        service, mock_emb, _ = make_service()
        with pytest.raises(InvalidQueryError):
            service.retrieve("")
        mock_emb.embed.assert_not_called()

    def test_empty_query_does_not_call_hybrid_search(self):
        service, mock_emb, mock_store = make_service()
        with pytest.raises(InvalidQueryError):
            service.retrieve("")
        mock_store.hybrid_search.assert_not_called()


# ---------------------------------------------------------------------------
# Query validation — too-long query (Req 6.6)
# ---------------------------------------------------------------------------

class TestInvalidQueryTooLong:
    """Query exceeding 4000 chars must raise InvalidQueryError before embedding."""

    def test_4001_chars_raises(self):
        service, mock_emb, _ = make_service()
        long_query = "a" * 4001
        with pytest.raises(InvalidQueryError):
            service.retrieve(long_query)

    def test_too_long_query_does_not_call_embed(self):
        service, mock_emb, _ = make_service()
        with pytest.raises(InvalidQueryError):
            service.retrieve("x" * 4001)
        mock_emb.embed.assert_not_called()

    def test_too_long_query_does_not_call_hybrid_search(self):
        service, _, mock_store = make_service()
        with pytest.raises(InvalidQueryError):
            service.retrieve("x" * 4001)
        mock_store.hybrid_search.assert_not_called()

    def test_exactly_4000_chars_is_valid(self):
        """Boundary: 4000 chars is the max allowed length — should NOT raise."""
        service, mock_emb, _ = make_service()
        service.retrieve("a" * 4000)
        mock_emb.embed.assert_called_once()

    def test_single_char_query_is_valid(self):
        """Boundary: minimum valid query is 1 character."""
        service, mock_emb, _ = make_service()
        service.retrieve("q")
        mock_emb.embed.assert_called_once()


# ---------------------------------------------------------------------------
# top_k validation (Req 6.6 / Req 6.3)
# ---------------------------------------------------------------------------

class TestInvalidTopK:
    """Out-of-range top_k must raise InvalidQueryError before embedding."""

    def test_top_k_zero_raises(self):
        service, mock_emb, _ = make_service()
        with pytest.raises(InvalidQueryError):
            service.retrieve("valid query", top_k=0)
        mock_emb.embed.assert_not_called()

    def test_top_k_negative_raises(self):
        service, mock_emb, _ = make_service()
        with pytest.raises(InvalidQueryError):
            service.retrieve("valid query", top_k=-1)
        mock_emb.embed.assert_not_called()

    def test_top_k_101_raises(self):
        service, mock_emb, _ = make_service()
        with pytest.raises(InvalidQueryError):
            service.retrieve("valid query", top_k=101)
        mock_emb.embed.assert_not_called()

    def test_top_k_1_is_valid(self):
        """Boundary: minimum valid top_k is 1."""
        service, mock_emb, _ = make_service()
        service.retrieve("valid query", top_k=1)
        mock_emb.embed.assert_called_once()

    def test_top_k_100_is_valid(self):
        """Boundary: maximum valid top_k is 100."""
        service, mock_emb, _ = make_service()
        service.retrieve("valid query", top_k=100)
        mock_emb.embed.assert_called_once()

    def test_top_k_default_is_5(self):
        """Default top_k of 5 is used when not specified."""
        service, _, mock_store = make_service()
        service.retrieve("valid query")
        mock_store.hybrid_search.assert_called_once()
        _, kwargs = mock_store.hybrid_search.call_args
        assert kwargs.get("top_k") == 5 or mock_store.hybrid_search.call_args[0][2] == 5


# ---------------------------------------------------------------------------
# Happy path — embedding is called once with the query (Req 6.2)
# ---------------------------------------------------------------------------

class TestEmbedding:
    """Valid query must result in exactly one embed call with the query text."""

    def test_embed_called_once(self):
        service, mock_emb, _ = make_service()
        service.retrieve("tell me about the document")
        mock_emb.embed.assert_called_once()

    def test_embed_called_with_query_list(self):
        """embed() must receive a single-element list containing the query."""
        service, mock_emb, _ = make_service()
        query = "what is retrieval augmented generation?"
        service.retrieve(query)
        mock_emb.embed.assert_called_once_with([query])

    def test_embed_not_called_for_empty_query(self):
        """Embedding is never called for invalid queries (Req 6.6)."""
        service, mock_emb, _ = make_service()
        with pytest.raises(InvalidQueryError):
            service.retrieve("")
        mock_emb.embed.assert_not_called()


# ---------------------------------------------------------------------------
# Happy path — hybrid_search is called once (Req 6.3)
# ---------------------------------------------------------------------------

class TestHybridSearch:
    """After embedding, hybrid_search must be called exactly once."""

    def test_hybrid_search_called_once(self):
        service, _, mock_store = make_service()
        service.retrieve("some query")
        mock_store.hybrid_search.assert_called_once()

    def test_hybrid_search_receives_dense_and_sparse(self):
        """The dense + sparse vectors from the embedding are passed to hybrid_search."""
        emb = make_embedding_result()
        service, _, mock_store = make_service(emb_result=emb)
        service.retrieve("query text", top_k=10)

        args, kwargs = mock_store.hybrid_search.call_args
        # Accept both positional and keyword call styles
        call_kwargs = {**kwargs}
        if args:
            param_names = ["dense", "sparse", "top_k"]
            for i, v in enumerate(args):
                call_kwargs[param_names[i]] = v

        assert call_kwargs["dense"] == emb.dense
        assert call_kwargs["sparse"] == emb.sparse
        assert call_kwargs["top_k"] == 10

    def test_hybrid_search_not_called_for_invalid_query(self):
        service, _, mock_store = make_service()
        with pytest.raises(InvalidQueryError):
            service.retrieve("x" * 5000)
        mock_store.hybrid_search.assert_not_called()


# ---------------------------------------------------------------------------
# Result assembly — candidates included with text, score, metadata (Req 6.4)
# ---------------------------------------------------------------------------

class TestResultAssembly:
    """Returned RetrievalResult must include text, score, and metadata (Req 6.4)."""

    def test_returns_retrieval_result(self):
        service, _, _ = make_service()
        result = service.retrieve("query")
        assert isinstance(result, RetrievalResult)

    def test_candidates_from_hybrid_search_are_in_result(self):
        cands = [
            make_candidate("passage one", 0.9),
            make_candidate("passage two", 0.7),
        ]
        service, _, _ = make_service(candidates=cands)
        result = service.retrieve("query")
        assert result.candidates == cands

    def test_candidate_has_text(self):
        cand = make_candidate("important passage", 0.8)
        service, _, _ = make_service(candidates=[cand])
        result = service.retrieve("query")
        assert result.candidates[0].text == "important passage"

    def test_candidate_has_score(self):
        cand = make_candidate("text", 0.654)
        service, _, _ = make_service(candidates=[cand])
        result = service.retrieve("query")
        assert result.candidates[0].score == pytest.approx(0.654)

    def test_candidate_has_metadata(self):
        meta = make_metadata(pdf_id="xyz", filename="book.pdf", page_number=42)
        cand = Candidate(text="content", score=0.5, metadata=meta)
        service, _, _ = make_service(candidates=[cand])
        result = service.retrieve("query")
        assert result.candidates[0].metadata.pdf_id == "xyz"
        assert result.candidates[0].metadata.filename == "book.pdf"
        assert result.candidates[0].metadata.page_number == 42

    def test_candidate_order_preserved(self):
        """Candidates are returned in the same order as hybrid_search (Req 6.3)."""
        cands = [
            make_candidate("first", 0.95),
            make_candidate("second", 0.80),
            make_candidate("third", 0.60),
        ]
        service, _, _ = make_service(candidates=cands)
        result = service.retrieve("query", top_k=3)
        assert [c.text for c in result.candidates] == ["first", "second", "third"]

    def test_error_field_is_none_on_success(self):
        service, _, _ = make_service()
        result = service.retrieve("valid query")
        assert result.error is None


# ---------------------------------------------------------------------------
# Empty result when store has no chunks (Req 6.5)
# ---------------------------------------------------------------------------

class TestEmptyResult:
    """When the Vector_Store returns no candidates, return an empty result (Req 6.5)."""

    def test_empty_candidates_list_when_store_returns_nothing(self):
        service, _, _ = make_service(candidates=[])
        result = service.retrieve("some query")
        assert result.candidates == []

    def test_result_is_still_retrieval_result_instance(self):
        service, _, _ = make_service(candidates=[])
        result = service.retrieve("some query")
        assert isinstance(result, RetrievalResult)

    def test_error_is_none_for_empty_result(self):
        service, _, _ = make_service(candidates=[])
        result = service.retrieve("some query")
        assert result.error is None


# ---------------------------------------------------------------------------
# Persistence / no-upload requirement (Req 5.7)
# ---------------------------------------------------------------------------

class TestPersistenceRequirement:
    """RetrievalService must use the injected VectorStore without requiring upload."""

    def test_service_uses_injected_store(self):
        """Service must call hybrid_search on the store it was given, not a new one."""
        mock_emb = make_mock_embedding_model()
        mock_store = make_mock_vector_store(candidates=[make_candidate()])
        service = RetrievalService(
            embedding_model=mock_emb,
            vector_store=mock_store,
        )
        service.retrieve("query about the corpus")
        # The injected store was consulted — not any separate upload path
        mock_store.hybrid_search.assert_called_once()

    def test_service_does_not_call_upsert(self):
        """Retrieval must NOT write to the store (no upsert, no ingestion)."""
        mock_emb = make_mock_embedding_model()
        mock_store = make_mock_vector_store()
        service = RetrievalService(
            embedding_model=mock_emb,
            vector_store=mock_store,
        )
        service.retrieve("another query")
        mock_store.upsert_chunk.assert_not_called()


# ---------------------------------------------------------------------------
# InvalidQueryError type
# ---------------------------------------------------------------------------

class TestInvalidQueryErrorType:
    def test_is_exception_subclass(self):
        err = InvalidQueryError("test message")
        assert isinstance(err, Exception)

    def test_carries_message(self):
        err = InvalidQueryError("query too long")
        assert "query too long" in str(err)


# ---------------------------------------------------------------------------
# RetrievalResult dataclass
# ---------------------------------------------------------------------------

class TestRetrievalResultDataclass:
    def test_default_candidates_is_empty_list(self):
        result = RetrievalResult()
        assert result.candidates == []

    def test_default_error_is_none(self):
        result = RetrievalResult()
        assert result.error is None

    def test_accepts_candidates_and_error(self):
        cands = [make_candidate()]
        result = RetrievalResult(candidates=cands, error="something went wrong")
        assert result.candidates == cands
        assert result.error == "something went wrong"

    def test_candidates_lists_are_independent(self):
        """Each RetrievalResult instance must have its own independent list."""
        r1 = RetrievalResult()
        r2 = RetrievalResult()
        r1.candidates.append(make_candidate())
        assert r2.candidates == [], "Mutation of one instance must not affect another"
