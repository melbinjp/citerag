"""
backend/reranker.py

Optional Reranker component for the RAG PDF Chatbot.

Re-orders a list of hybrid-search Candidate chunks by relevance to the query,
using a lightweight token-overlap scoring approach that requires no GPU or
heavy cross-encoder dependencies.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable

from backend.data_models import Candidate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lexical relevance scoring (lightweight, CPU-only)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lower-case, split on non-alphanumeric boundaries, return unique tokens.

    Intentionally simple — suitable for MVP token-overlap scoring without any
    external NLP library dependency.
    """
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _token_overlap_score(query: str, candidate_text: str) -> float:
    """Return the Jaccard-style token overlap score between *query* and *candidate_text*.

    Score is the number of shared tokens divided by the total unique tokens
    across both strings.  Returns 0.0 when both token sets are empty.

    This is a deterministic, parameter-free relevance proxy that:
    - produces higher scores for candidates that share more query keywords;
    - is scale-free (always in [0.0, 1.0]);
    - requires no external models.
    """
    q_tokens = _tokenize(query)
    c_tokens = _tokenize(candidate_text)
    union = q_tokens | c_tokens
    if not union:
        return 0.0
    return len(q_tokens & c_tokens) / len(union)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class Reranker:
    """Optional stage that re-orders Hybrid_Search candidates by relevance.

    Behaviour summary
    -----------------
    - **disabled** (``enabled=False``): :meth:`rerank` returns the input list
      unchanged, preserving the original Hybrid_Search order (Req 7.2).
    - **enabled** (``enabled=True``): candidates are scored with the
      *scoring_fn* (default: token-overlap), then sorted by descending score;
      ties are broken by original Hybrid_Search rank (stable, Req 7.4).
      The result is always a permutation of the input — same count, same
      objects, no metadata mutation (Req 7.3).
    - **error / timeout**: any exception raised by the scoring function, or
      expiry of *timeout_s* inside :meth:`rerank`, causes the method to return
      the **input order** (fallback, Req 7.5).  The exception is logged at
      WARNING level; callers must not re-raise.

    Parameters
    ----------
    enabled:
        When ``False`` the reranker is a pass-through (Req 7.2).
    scoring_fn:
        Callable ``(query: str, candidate_text: str) -> float``.  Must be
        deterministic and raise no exceptions for valid string inputs.
        Defaults to :func:`_token_overlap_score`.
    """

    def __init__(
        self,
        enabled: bool = False,
        scoring_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self._enabled = enabled
        self._scoring_fn: Callable[[str, str], float] = (
            scoring_fn if scoring_fn is not None else _token_overlap_score
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """``True`` when the reranker will actively reorder candidates."""
        return self._enabled

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        timeout_s: float = 2.0,
    ) -> list[Candidate]:
        """Re-order *candidates* by relevance to *query*.

        Parameters
        ----------
        query:
            The user query string.
        candidates:
            Ordered list of :class:`~backend.data_models.Candidate` objects
            as produced by :class:`~backend.retrieval.RetrievalService`
            (i.e. Hybrid_Search order, index 0 = highest fused rank).
        timeout_s:
            Maximum wall-clock seconds allowed for the scoring step.  On
            expiry the method falls back to the input order (Req 7.5).

        Returns
        -------
        list[Candidate]
            A permutation of the input with identical objects (text and
            :class:`~backend.data_models.ChunkMetadata` preserved, Req 7.3).
            Always the same length as the input.

            - When ``enabled=False``: identical to the input list (Req 7.2).
            - When ``enabled=True`` and scoring succeeds: sorted by descending
              relevance score; ties keep original Hybrid_Search order (Req 7.4).
            - On any error or timeout: identical to the input list (Req 7.5).
        """
        # Pass-through when disabled (Req 7.2)
        if not self._enabled:
            return list(candidates)

        # Nothing to reorder
        if not candidates:
            return []

        # Run scoring with a timeout using a thread (Req 7.5)
        try:
            scored = self._score_with_timeout(query, candidates, timeout_s)
        except Exception as exc:  # noqa: BLE001 — intentional broad catch for fallback
            logger.warning(
                "Reranker error — falling back to fused order. reason=%s",
                exc,
            )
            return list(candidates)

        # Sort: descending score, then ascending original index (stable tie-break, Req 7.4)
        scored.sort(key=lambda item: (-item[1], item[0]))

        # Return only the candidates, preserving original objects (Req 7.3)
        return [cand for _, _, cand in scored]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        query: str,
        candidates: list[Candidate],
    ) -> list[tuple[int, float, Candidate]]:
        """Score every candidate and return ``(original_index, score, candidate)`` tuples.

        Raises whatever exception the scoring function raises so that
        :meth:`rerank` can catch it and fall back to fused order.
        """
        return [
            (idx, self._scoring_fn(query, cand.text), cand)
            for idx, cand in enumerate(candidates)
        ]

    def _score_with_timeout(
        self,
        query: str,
        candidates: list[Candidate],
        timeout_s: float,
    ) -> list[tuple[int, float, Candidate]]:
        """Run :meth:`_score_candidates` in a thread pool with a wall-clock timeout.

        Raises
        ------
        TimeoutError
            When the scoring does not complete within *timeout_s* seconds.
        Exception
            Any exception raised by the scoring function propagates here.
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._score_candidates, query, candidates)
            try:
                return future.result(timeout=timeout_s)
            except FuturesTimeoutError as exc:
                # Cancel if still running; the thread may linger until the
                # scoring function yields, but that is acceptable for a
                # CPU-bound lightweight scorer.
                future.cancel()
                raise TimeoutError(
                    f"Reranker scoring exceeded timeout of {timeout_s}s"
                ) from exc
