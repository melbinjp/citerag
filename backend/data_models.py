"""
backend/data_models.py

Core frozen dataclasses shared across the RAG pipeline.

These are the canonical definitions of ChunkMetadata, Chunk, Candidate,
Citation, Turn, GenerationResult, SparseVector, and EmbeddingResult.
IngestionErrorEntry lives in backend/ingestion/error_log.py (its own module
for historical reasons) and is re-exported here for convenience.

Requirements: 4.1, 4.2, 4.3, 4.4
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Vector representations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SparseVector:
    """Sparse lexical vector produced by the Embedding_Model (bge-m3).

    Attributes:
        indices: Term ids (non-zero positions in the high-dimensional space).
        values:  Corresponding term weights (BM25-style lexical scores).
    """

    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class EmbeddingResult:
    """Dense + sparse embedding pair for a single text, returned by EmbeddingModel.

    Attributes:
        dense:  Dense semantic vector, length = model dimension (1024 for bge-m3).
        sparse: Sparse lexical vector (BM25-style weights).
    """

    dense: list[float]      # length = 1024 for BAAI/bge-m3
    sparse: SparseVector


# ---------------------------------------------------------------------------
# Chunk and metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkMetadata:
    """Per-chunk provenance attached by the Chunker (Requirement 3.3, 3.4).

    Attributes:
        pdf_id:         Stable identifier derived from the PDF file path/content.
        filename:       Original PDF filename (basename).
        page_number:    1-based page number of the page that contributes the
                        chunk's first token.  For cross-page chunks this is the
                        page of the first token (Req 3.4).
        chunk_position: 0-based ordinal of this chunk within the PDF.
        language:       ISO 639-1 two-letter lowercase code (e.g. ``"en"``) or
                        ``"und"`` when language detection was inconclusive.
    """

    pdf_id: str
    filename: str
    page_number: int       # 1-based; page of first token for cross-page chunks
    chunk_position: int    # 0-based ordinal within the PDF
    language: str          # ISO 639-1 or "und"


@dataclass(frozen=True)
class Chunk:
    """A single token-bounded text passage with its provenance metadata.

    Produced by the Chunker and consumed by the Ingestion_Pipeline for
    embedding and persistence (Requirement 3).
    """

    text: str
    metadata: ChunkMetadata


# ---------------------------------------------------------------------------
# Retrieval result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """A retrieved chunk with its fused relevance score.

    Returned by the Retrieval_Service after hybrid search (and optionally
    after reranking).

    Attributes:
        text:     Raw chunk text as stored in the Vector_Store.
        score:    Fused RRF score (or rerank score when reranking is enabled).
        metadata: Full ChunkMetadata associated with this chunk.
    """

    text: str
    score: float           # fused RRF score or rerank score
    metadata: ChunkMetadata


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Citation:
    """Provenance reference attached to a generated answer.

    Attributes:
        filename: Source PDF filename.
        pages:    Sorted list of unique page numbers from retrieved chunks of
                  this PDF that contributed to the answer.
    """

    filename: str
    pages: list[int]       # sorted, unique page numbers


@dataclass(frozen=True)
class Turn:
    """A single question-answer exchange in a Conversation_Session.

    Attributes:
        question: The user's question text.
        answer:   The assistant's answer text.
    """

    question: str
    answer: str


@dataclass
class GenerationResult:
    """Output of a single Answer_Generator invocation.

    Attributes:
        answer:    Generated answer text (empty string on error).
        citations: Structured citations (empty list on error or empty context).
        error:     Error description when generation failed; ``None`` otherwise.
    """

    answer: str
    citations: list[Citation]
    error: str | None = None


# ---------------------------------------------------------------------------
# Re-export IngestionErrorEntry for convenience
# (canonical definition lives in backend/ingestion/error_log.py)
# ---------------------------------------------------------------------------

try:
    from backend.ingestion.error_log import IngestionErrorEntry  # noqa: E402
except ImportError:
    from ingestion.error_log import IngestionErrorEntry  # noqa: E402
