"""Embedding model wrapper for BAAI/bge-m3.

Loads the model once on construction and produces both a dense semantic vector
and a sparse lexical vector for each input text in a single forward pass.

Requirements: 4.1, 4.2
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SparseVector:
    """Sparse lexical vector produced by bge-m3.

    Attributes:
        indices: Term ids (token ids) with non-zero weights.
        values:  Corresponding term weights.
    """

    indices: list[int]
    values: list[float]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"SparseVector: indices ({len(self.indices)}) and values "
                f"({len(self.values)}) must have the same length."
            )


@dataclass(frozen=True)
class EmbeddingResult:
    """Combined dense + sparse embedding for a single input text.

    Attributes:
        dense:  Dense semantic vector (length 1024 for BAAI/bge-m3).
        sparse: Sparse lexical vector (indices + weights).
    """

    dense: list[float]
    sparse: SparseVector


# ---------------------------------------------------------------------------
# Embedding model wrapper
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """Wraps BAAI/bge-m3; returns dense and sparse vectors in one forward pass.

    The underlying ``BGEM3FlagModel`` is loaded **once** during ``__init__``.
    Subsequent calls to :meth:`embed` re-use the same model instance.

    Args:
        model_name: HuggingFace model id.  Defaults to ``"BAAI/bge-m3"``.

    Example::

        model = EmbeddingModel()
        results = model.embed(["hello world", "another text"])
        # results[0].dense  -> list of 1024 floats
        # results[0].sparse -> SparseVector(indices=[...], values=[...])
    """

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self._model_name = model_name
        logger.info("Loading embedding model '%s' …", model_name)
        self._model = self._load_model(model_name)
        logger.info("Embedding model '%s' loaded.", model_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(model_name: str):
        """Load and return a ``BGEM3FlagModel`` instance.

        Separated into a static method to make it easy to mock in tests
        without downloading the real model weights.
        """
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is required.  Install it with: pip install FlagEmbedding"
            ) from exc

        return BGEM3FlagModel(
            model_name,
            use_fp16=False,   # CPU-only deployment – keep fp32 for correctness
        )

    @staticmethod
    def _extract_sparse(raw_sparse: dict) -> SparseVector:
        """Convert the raw sparse output from BGEM3FlagModel to a SparseVector.

        BGEM3FlagModel.encode with ``return_sparse=True`` returns a list of
        dicts (one per text), where each dict maps term-id (int) to weight
        (float).  Some versions return the dict directly, others wrap it under
        a ``"lexical_weights"`` key.

        Args:
            raw_sparse: The per-text sparse dict from the model output.

        Returns:
            A :class:`SparseVector` with sorted indices for determinism.
        """
        if isinstance(raw_sparse, dict):
            sparse_dict = raw_sparse
        else:
            # Fallback: try the 'lexical_weights' attribute convention
            raise TypeError(
                f"Unexpected sparse output type {type(raw_sparse).__name__}; "
                "expected a dict mapping term-id -> weight."
            )

        sorted_items = sorted(sparse_dict.items())  # sort by term-id for determinism
        indices = [int(k) for k, _ in sorted_items]
        values = [float(v) for _, v in sorted_items]
        return SparseVector(indices=indices, values=values)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a list of texts in a **single forward pass**.

        Returns exactly one :class:`EmbeddingResult` per input text, in the
        same order as *texts*.  Each result contains both the dense (1024-dim
        for bge-m3) and sparse vectors.

        Args:
            texts: Input strings to embed.  May be empty, in which case an
                   empty list is returned immediately without calling the model.

        Returns:
            A list of :class:`EmbeddingResult` objects, one per input text,
            preserving the input order.

        Raises:
            ValueError: If the model returns a different number of embeddings
                        than input texts.
        """
        if not texts:
            return []

        output = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=len(texts),  # single pass over the whole input list
        )

        dense_embeddings = output["dense_vecs"]    # shape: (n, 1024) – numpy array
        sparse_embeddings = output["lexical_weights"]  # list of dicts, one per text

        n_texts = len(texts)
        n_dense = len(dense_embeddings)
        n_sparse = len(sparse_embeddings)

        if n_dense != n_texts:
            raise ValueError(
                f"Model returned {n_dense} dense vectors for {n_texts} input texts."
            )
        if n_sparse != n_texts:
            raise ValueError(
                f"Model returned {n_sparse} sparse vectors for {n_texts} input texts."
            )

        results: list[EmbeddingResult] = []
        for i in range(n_texts):
            dense_vec = [float(x) for x in dense_embeddings[i]]
            sparse_vec = self._extract_sparse(sparse_embeddings[i])
            results.append(EmbeddingResult(dense=dense_vec, sparse=sparse_vec))

        return results
