"""
backend/tests/test_reranker.py

Unit tests for backend/reranker.py

Tests cover:
- Pass-through when disabled (Req 7.2)
- Descending-score ordering when enabled (Req 7.1)
- Stable tie-breaking by original rank (Req 7.4)
- Metadata-preserving permutation — same objects, same count (Req 7.3)
- Fallback to fused order on exception (Req 7.5)
- Fallback to fused order on timeout (Req 7.5)
- Empty candidate list edge case
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from backend.data_models import Candidate, ChunkMetadata
from backend.reranker import Reranker, _token_overlap_score, _tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_metadata(page: int = 1, position: int = 0) -> ChunkMetadata:
    return ChunkMetadata(
        pdf_id="doc1",
        filename="test.pdf",
        page_number=page,
        chunk_position=position,
        language="en",
    )


def make_candidate(text: str, score: float = 0.5, page: int = 1, position: int = 0) -> Candidate:
    return Candidate(text=text, score=score, metadata=make_metadata(page, position))


# ---------------------------------------------------------------------------
# _tokenize and _token_overlap_score unit tests
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_words(self):
        result = _tokenize("Hello World")
        assert result == {"hello", "world"}

    def test_punctuation_stripped(self):
        result = _tokenize("foo, bar! baz.")
        assert result == {"foo", "bar", "baz"}

    def test_numbers_kept(self):
        assert "42" in _tokenize("page 42")

    def test_empty_string(self):
        assert _tokenize("") == set()

    def test_uniqueness(self):
        # Repeated tokens appear once
        assert _tokenize("the the the") == {"the"}


class TestTokenOverlapScore:
    def test_identical_texts(self):
        score = _token_overlap_score("machine learning", "machine learning")
        assert score == 1.0

    def test_no_overlap(self):
        score = _token_overlap_score("apple orange", "cat dog")
        assert score == 0.0

    def test_partial_overlap(self):
        score = _token_overlap_score("machine learning model", "deep learning networks")
        # shared: {"learning"} = 1, union = {"machine","learning","model","deep","networks"} = 5
        assert score == pytest.approx(1 / 5)

    def test_empty_query_and_text(self):
        assert _token_overlap_score("", "") == 0.0

    def test_empty_query(self):
        # no shared tokens
        assert _token_overlap_score("", "some text") == 0.0

    def test_score_range(self):
        for q, c in [("foo", "bar"), ("foo", "foo"), ("a b c", "b c d")]:
            score = _token_overlap_score(q, c)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Reranker disabled (pass-through)
# ---------------------------------------------------------------------------

class TestRerankerDisabled:
    def test_returns_same_order(self):
        reranker = Reranker(enabled=False)
        candidates = [
            make_candidate("alpha text", score=0.9, position=0),
            make_candidate("beta text", score=0.7, position=1),
            make_candidate("gamma text", score=0.5, position=2),
        ]
        result = reranker.rerank("query", candidates, timeout_s=2.0)
        assert result == candidates

    def test_returns_copy_not_same_list(self):
        reranker = Reranker(enabled=False)
        candidates = [make_candidate("text")]
        result = reranker.rerank("query", candidates)
        assert result is not candidates

    def test_empty_list(self):
        reranker = Reranker(enabled=False)
        assert reranker.rerank("query", []) == []

    def test_enabled_property_false(self):
        assert Reranker(enabled=False).enabled is False


# ---------------------------------------------------------------------------
# Reranker enabled — ordering
# ---------------------------------------------------------------------------

class TestRerankerEnabled:
    def test_sorts_by_descending_score(self):
        """Req 7.1: reranked output ordered by descending relevance."""
        reranker = Reranker(enabled=True)
        # Candidate texts crafted so overlap with query "machine learning" varies
        candidates = [
            make_candidate("deep neural networks", position=0),    # low overlap
            make_candidate("machine learning algorithms", position=1),  # high overlap
            make_candidate("learning systems", position=2),         # medium overlap
        ]
        result = reranker.rerank("machine learning", candidates)
        scores = [_token_overlap_score("machine learning", c.text) for c in result]
        assert scores == sorted(scores, reverse=True), "Result not in descending score order"

    def test_same_candidates_returned(self):
        """Req 7.3: result is a permutation — same objects, same count."""
        reranker = Reranker(enabled=True)
        candidates = [
            make_candidate("text one", position=0),
            make_candidate("text two", position=1),
            make_candidate("text three", position=2),
        ]
        result = reranker.rerank("text", candidates)
        assert len(result) == len(candidates)
        assert set(id(c) for c in result) == set(id(c) for c in candidates)

    def test_metadata_preserved(self):
        """Req 7.3: chunk text and ChunkMetadata are unchanged after reranking."""
        reranker = Reranker(enabled=True)
        meta1 = make_metadata(page=1, position=0)
        meta2 = make_metadata(page=5, position=1)
        candidates = [
            Candidate(text="alpha query match", score=0.8, metadata=meta1),
            Candidate(text="unrelated content", score=0.9, metadata=meta2),
        ]
        result = reranker.rerank("alpha query", candidates)
        # Verify the same ChunkMetadata objects survive reranking unchanged
        result_metas = {c.metadata for c in result}
        assert meta1 in result_metas
        assert meta2 in result_metas

    def test_empty_list(self):
        reranker = Reranker(enabled=True)
        assert reranker.rerank("query", []) == []

    def test_enabled_property_true(self):
        assert Reranker(enabled=True).enabled is True

    def test_single_candidate(self):
        reranker = Reranker(enabled=True)
        cand = make_candidate("solo text")
        result = reranker.rerank("solo", [cand])
        assert result == [cand]


# ---------------------------------------------------------------------------
# Tie-breaking by original rank (Req 7.4)
# ---------------------------------------------------------------------------

class TestTieBreaking:
    def test_equal_scores_preserve_original_order(self):
        """Req 7.4: ties broken by original Hybrid_Search rank (lower index first)."""
        # Use a scoring function that returns the same score for every candidate
        constant_scorer = lambda q, text: 0.5  # noqa: E731
        reranker = Reranker(enabled=True, scoring_fn=constant_scorer)
        candidates = [
            make_candidate("first", position=0),
            make_candidate("second", position=1),
            make_candidate("third", position=2),
        ]
        result = reranker.rerank("query", candidates)
        # All scores equal → original order must be preserved
        assert result == candidates

    def test_partial_ties(self):
        """Candidates with same score keep original relative order; higher-score comes first."""
        scores_by_text = {
            "alpha": 0.9,
            "beta": 0.5,
            "gamma": 0.5,   # tied with beta; beta was before gamma → beta stays before gamma
            "delta": 0.3,
        }
        scorer = lambda q, text: scores_by_text.get(text.split()[0], 0.0)  # noqa: E731
        reranker = Reranker(enabled=True, scoring_fn=scorer)

        candidates = [
            make_candidate("delta text", position=0),
            make_candidate("beta text", position=1),
            make_candidate("gamma text", position=2),
            make_candidate("alpha text", position=3),
        ]
        result = reranker.rerank("query", candidates)
        texts = [c.text.split()[0] for c in result]
        assert texts[0] == "alpha"               # highest score first
        # beta (original idx 1) and gamma (original idx 2) tied → beta before gamma
        beta_pos = texts.index("beta")
        gamma_pos = texts.index("gamma")
        assert beta_pos < gamma_pos
        assert texts[-1] == "delta"              # lowest score last


# ---------------------------------------------------------------------------
# Fallback on error / timeout (Req 7.5)
# ---------------------------------------------------------------------------

class TestFallback:
    def test_fallback_on_scoring_exception(self):
        """Req 7.5: exception from scoring_fn → return input order."""
        def exploding_scorer(query: str, text: str) -> float:
            raise RuntimeError("scoring service unavailable")

        reranker = Reranker(enabled=True, scoring_fn=exploding_scorer)
        candidates = [
            make_candidate("first", position=0),
            make_candidate("second", position=1),
        ]
        result = reranker.rerank("query", candidates)
        assert result == candidates

    def test_fallback_on_timeout(self):
        """Req 7.5: scoring exceeding timeout_s → return input order."""
        def slow_scorer(query: str, text: str) -> float:
            time.sleep(5)   # far exceeds timeout_s=0.05
            return 1.0

        reranker = Reranker(enabled=True, scoring_fn=slow_scorer)
        candidates = [
            make_candidate("first", position=0),
            make_candidate("second", position=1),
        ]
        result = reranker.rerank("query", candidates, timeout_s=0.05)
        assert result == candidates

    def test_fallback_preserves_all_candidates(self):
        """Even on fallback, all candidates are returned with no additions or drops."""
        def exploding_scorer(query: str, text: str) -> float:
            raise ValueError("boom")

        reranker = Reranker(enabled=True, scoring_fn=exploding_scorer)
        candidates = [make_candidate(f"text {i}", position=i) for i in range(5)]
        result = reranker.rerank("query", candidates)
        assert len(result) == 5
        assert result == candidates


# ---------------------------------------------------------------------------
# Custom scoring function
# ---------------------------------------------------------------------------

class TestCustomScoringFn:
    def test_custom_scorer_used(self):
        """Reranker delegates to the supplied scoring_fn."""
        call_log: list[tuple[str, str]] = []

        def recording_scorer(query: str, text: str) -> float:
            call_log.append((query, text))
            return float(len(text))   # longer texts rank higher

        reranker = Reranker(enabled=True, scoring_fn=recording_scorer)
        candidates = [
            make_candidate("short", position=0),
            make_candidate("a much longer candidate text here", position=1),
        ]
        result = reranker.rerank("test query", candidates)

        assert len(call_log) == 2
        assert all(q == "test query" for q, _ in call_log)
        # Longer text should be ranked first
        assert result[0].text == "a much longer candidate text here"
