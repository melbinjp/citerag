"""
backend/retrieval.py

RetrievalService — validates queries, embeds them, performs hybrid search,
and assembles ranked candidate results.

Requirements: 5.7, 6.2, 6.3, 6.4, 6.5, 6.6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backend.data_models import Candidate
from backend.embedding import EmbeddingModel
from backend.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class InvalidQueryError(Exception):
    """Raised by RetrievalService when a query fails validation.

    Raised *before* any call to the Embedding_Model, so no embedding work
    is ever done for an invalid query (Requirement 6.6).
    """


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Output of a single :meth:`RetrievalService.retrieve` call.

    Attributes:
        candidates: Ranked list of retrieved chunks, ordered by descending
                    fused RRF score.  Empty when the Vector_Store contains no
                    chunks for the requested scope (Requirement 6.5).
        error:      Human-readable error description when retrieval failed for
                    a reason other than an invalid query (e.g. a store error).
                    ``None`` on success.
    """

    candidates: list[Candidate] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# RetrievalService
# ---------------------------------------------------------------------------

class RetrievalService:
    """Validates queries, embeds them, and performs hybrid search.

    The service is a thin orchestration layer between the Embedding_Model and
    the VectorStore.  It does *not* own or modify any state in those
    dependencies — it calls their public APIs and assembles the result.

    Parameters
    ----------
    embedding_model:
        An :class:`~backend.embedding.EmbeddingModel` instance used to embed
        the validated query into dense + sparse vectors (Requirement 6.2).
    vector_store:
        A :class:`~backend.vector_store.VectorStore` instance whose
        :meth:`~backend.vector_store.VectorStore.hybrid_search` method is
        called once per valid query (Requirement 6.3).

    Class-level constants
    ---------------------
    MAX_QUERY_CHARS:
        Upper bound on query length (inclusive).  Queries that exceed this
        limit are rejected before embedding (Requirement 6.6).
    DEFAULT_TOP_K:
        Default number of candidates returned when *top_k* is not specified
        (Requirement 6.3).
    MAX_TOP_K:
        Upper bound on *top_k* (inclusive).

    Requirements: 5.7, 6.2, 6.3, 6.4, 6.5, 6.6
    """

    MAX_QUERY_CHARS: int = 4000
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 100

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_store = vector_store

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> RetrievalResult:
        """Validate, embed, search, and return ranked candidates.

        Validation happens first, before any embedding work, so that invalid
        queries never incur embedding cost (Requirement 6.6).

        Parameters
        ----------
        query:
            User query string.  Must be between 1 and 4000 characters
            (inclusive).
        top_k:
            Maximum number of candidates to return.  Must be between 1 and
            100 (inclusive).  Defaults to 5.

        Returns
        -------
        RetrievalResult
            On success: a :class:`RetrievalResult` whose ``candidates`` list
            contains at most *top_k* :class:`~backend.data_models.Candidate`
            objects ordered by descending fused RRF score (Requirement 6.3).
            Each candidate includes its chunk text, fused relevance score, and
            :class:`~backend.data_models.ChunkMetadata` (Requirement 6.4).
            When the Vector_Store contains no chunks for the scope, returns an
            empty ``candidates`` list (Requirement 6.5).

        Raises
        ------
        InvalidQueryError
            When ``query`` is empty, exceeds ``MAX_QUERY_CHARS`` characters, or
            ``top_k`` is outside the range 1..100.  The Embedding_Model is
            never called in these cases (Requirement 6.6).
        """
        # ------------------------------------------------------------------
        # 1. Validate — must come before any embedding call (Req 6.6)
        # ------------------------------------------------------------------
        self._validate(query, top_k)

        # ------------------------------------------------------------------
        # 2. Embed query: dense + sparse in a single forward pass (Req 6.2)
        # ------------------------------------------------------------------
        logger.debug("Embedding query (len=%d, top_k=%d).", len(query), top_k)
        embedding_results = self._embedding_model.embed([query])
        query_emb = embedding_results[0]

        # ------------------------------------------------------------------
        # 3. Single hybrid search fused via RRF (Req 6.3)
        # ------------------------------------------------------------------
        logger.debug("Performing hybrid search (top_k=%d).", top_k)
        candidates = self._vector_store.hybrid_search(
            dense=query_emb.dense,
            sparse=query_emb.sparse,
            top_k=top_k,
        )

        # ------------------------------------------------------------------
        # 4. Assemble result (Req 6.4, 6.5)
        # ------------------------------------------------------------------
        # hybrid_search already returns Candidate objects with text, score,
        # and ChunkMetadata populated (Req 6.4).  An empty list satisfies
        # Req 6.5 (no chunks for scope → empty result set).
        logger.debug("Retrieval complete: %d candidates returned.", len(candidates))
        return RetrievalResult(candidates=candidates)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate(self, query: str, top_k: int) -> None:
        """Raise :exc:`InvalidQueryError` for any invalid input.

        Called unconditionally at the start of :meth:`retrieve` so that
        the Embedding_Model is never invoked for invalid queries (Req 6.6).

        Parameters
        ----------
        query:
            Query string to validate.
        top_k:
            Requested number of candidates to validate.

        Raises
        ------
        InvalidQueryError
            When ``query`` is empty or longer than ``MAX_QUERY_CHARS``, or
            when ``top_k`` is outside 1..``MAX_TOP_K``.
        """
        if not query:
            raise InvalidQueryError(
                "Query must not be empty (Requirement 6.6)."
            )
        if len(query) > self.MAX_QUERY_CHARS:
            raise InvalidQueryError(
                f"Query exceeds the maximum allowed length of "
                f"{self.MAX_QUERY_CHARS} characters "
                f"(received {len(query)} characters). "
                "Embedding was NOT called (Requirement 6.6)."
            )
        if top_k < 1 or top_k > self.MAX_TOP_K:
            raise InvalidQueryError(
                f"top_k must be between 1 and {self.MAX_TOP_K} inclusive "
                f"(received {top_k})."
            )
