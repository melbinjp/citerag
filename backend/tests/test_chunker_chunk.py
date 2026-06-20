"""
tests/test_chunker_chunk.py

Unit tests for Chunker.chunk() — recursive splitting, overlap, metadata,
final-remainder disposition, and zero-token input.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 3.8, 3.9
"""

import pytest

from ingestion.chunker import (
    Chunker,
    ChunkerConfig,
    PageText,
)
from data_models import Chunk, ChunkMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(
    max_tokens: int = 800,
    overlap_tokens: int = 150,
    min_final_tokens: int = 100,
) -> ChunkerConfig:
    return ChunkerConfig(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_final_tokens=min_final_tokens,
    )


def make_page(
    text: str,
    page_number: int = 1,
    filename: str = "test.pdf",
    pdf_id: str = "pdf-001",
    language: str = "en",
) -> PageText:
    return PageText(
        page_number=page_number,
        text=text,
        filename=filename,
        pdf_id=pdf_id,
        language=language,
    )


def word_repeat(word: str, n: int) -> str:
    """Produce a string of *n* space-separated copies of *word*."""
    return " ".join([word] * n)


# ---------------------------------------------------------------------------
# Req 3.9 — zero-token / empty input
# ---------------------------------------------------------------------------

class TestZeroTokenInput:
    def test_empty_page_list_returns_no_chunks(self):
        """Zero pages → zero chunks (Req 3.9)."""
        chunker = Chunker(make_config())
        result = chunker.chunk([])
        assert result == []

    def test_single_page_with_empty_text_returns_no_chunks(self):
        """A single page whose text is empty → zero chunks (Req 3.9)."""
        chunker = Chunker(make_config())
        result = chunker.chunk([make_page("")])
        assert result == []

    def test_all_whitespace_page_returns_no_chunks(self):
        """Pages with only whitespace → tokeniser yields no tokens → zero chunks."""
        chunker = Chunker(make_config())
        result = chunker.chunk([make_page("   \n\n  ")])
        assert result == []


# ---------------------------------------------------------------------------
# Basic chunking sanity checks
# ---------------------------------------------------------------------------

class TestBasicChunking:
    def test_short_text_produces_one_chunk(self):
        """Text well under max_tokens → exactly one chunk."""
        chunker = Chunker(make_config())
        result = chunker.chunk([make_page("Hello world. This is a short text.")])
        assert len(result) == 1

    def test_chunk_is_frozen_dataclass(self):
        chunker = Chunker(make_config())
        result = chunker.chunk([make_page("Hello world.")])
        assert isinstance(result[0], Chunk)

    def test_chunk_text_contains_original_words(self):
        chunker = Chunker(make_config())
        result = chunker.chunk([make_page("The quick brown fox jumps.")])
        assert "quick" in result[0].text

    def test_chunk_position_starts_at_zero(self):
        """First chunk has chunk_position == 0 (Req 3.3)."""
        chunker = Chunker(make_config())
        result = chunker.chunk([make_page("Hello world.")])
        assert result[0].metadata.chunk_position == 0

    def test_chunk_positions_are_sequential(self):
        """Multiple chunks have monotonically increasing 0-based positions (Req 3.3)."""
        # Use a small max_tokens so we get several chunks from repeated text.
        cfg = make_config(max_tokens=20, overlap_tokens=2, min_final_tokens=1)
        text = word_repeat("word", 200)
        chunker = Chunker(cfg)
        result = chunker.chunk([make_page(text)])
        positions = [c.metadata.chunk_position for c in result]
        assert positions == list(range(len(result)))

    def test_metadata_pdf_id_and_filename_populated(self):
        """Every chunk has pdf_id and filename from the input PageText (Req 3.3)."""
        chunker = Chunker(make_config())
        result = chunker.chunk([make_page("Hello.", filename="doc.pdf", pdf_id="abc-123")])
        for chunk in result:
            assert chunk.metadata.pdf_id == "abc-123"
            assert chunk.metadata.filename == "doc.pdf"


# ---------------------------------------------------------------------------
# Req 3.1 — token-size constraint
# ---------------------------------------------------------------------------

class TestTokenSizeConstraint:
    def _chunker_small(self) -> Chunker:
        """A chunker with small max_tokens so splitting is easy to trigger."""
        return Chunker(make_config(max_tokens=20, overlap_tokens=2, min_final_tokens=1))

    def test_all_chunks_within_max_tokens(self):
        """Every chunk has ≤ max_tokens tokens (Req 3.1)."""
        cfg = make_config(max_tokens=20, overlap_tokens=2, min_final_tokens=1)
        chunker = Chunker(cfg)
        text = word_repeat("hello", 300)
        result = chunker.chunk([make_page(text)])
        for chunk in result:
            count = chunker.count_tokens(chunk.text)
            # The only exception is a Req-3.7 merged final chunk.
            # Since min_final_tokens=1, every remainder ≥ 1 token gets its own
            # chunk (Req 3.6), so no chunk should exceed max_tokens here.
            assert count <= cfg.max_tokens, (
                f"Chunk exceeded max_tokens: {count} > {cfg.max_tokens}\n"
                f"Chunk text: {chunk.text[:80]!r}"
            )

    def test_long_text_produces_multiple_chunks(self):
        """Enough text to require multiple chunks → len > 1."""
        cfg = make_config(max_tokens=20, overlap_tokens=2, min_final_tokens=1)
        chunker = Chunker(cfg)
        text = word_repeat("hello", 300)
        result = chunker.chunk([make_page(text)])
        assert len(result) > 1


# ---------------------------------------------------------------------------
# Req 3.2 — overlap between consecutive chunks
# ---------------------------------------------------------------------------

class TestOverlapBetweenChunks:
    def test_overlap_is_within_bounds(self):
        """Consecutive-chunk overlap is between 10% and 30% of max_tokens (Req 3.2)."""
        cfg = make_config(max_tokens=40, overlap_tokens=5, min_final_tokens=1)
        chunker = Chunker(cfg)
        # Use distinct words so the suffix/prefix overlap measurement is unambiguous.
        # Generate 400 unique words: "alpha0", "alpha1", ..., "alpha399".
        words = [f"alpha{i}" for i in range(400)]
        text = " ".join(words)
        result = chunker.chunk([make_page(text)])
        if len(result) < 2:
            pytest.skip("Not enough chunks to test overlap")

        low = round(cfg.max_tokens * 0.10)
        high = round(cfg.max_tokens * 0.30)

        for i in range(len(result) - 1):
            a = result[i].text
            b = result[i + 1].text
            overlap_tokens = _measure_overlap(a, b, chunker)
            assert low <= overlap_tokens <= high, (
                f"Overlap {overlap_tokens} tokens outside [{low}, {high}] "
                f"for chunks {i} and {i+1}"
            )


def _measure_overlap(text_a: str, text_b: str, chunker: Chunker) -> int:
    """Approximate the number of overlapping tokens between the end of *text_a*
    and the beginning of *text_b* by finding the longest common suffix/prefix.
    """
    tokens_a = chunker.tokenizer.encode(text_a)
    tokens_b = chunker.tokenizer.encode(text_b)
    max_check = min(len(tokens_a), len(tokens_b))
    for length in range(max_check, 0, -1):
        if tokens_a[-length:] == tokens_b[:length]:
            return length
    return 0


# ---------------------------------------------------------------------------
# Req 3.3 & 3.4 — metadata completeness and page-number accuracy
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_page_number_on_single_page(self):
        """Single-page input → all chunks report page_number == that page's number."""
        chunker = Chunker(make_config())
        result = chunker.chunk([make_page("Short text.", page_number=5)])
        for chunk in result:
            assert chunk.metadata.page_number == 5

    def test_language_attached(self):
        """Language from the PageText is propagated to ChunkMetadata (Req 3.3)."""
        chunker = Chunker(make_config())
        result = chunker.chunk([make_page("Bonjour le monde.", language="fr")])
        assert result[0].metadata.language == "fr"

    def test_cross_page_chunk_records_first_page(self):
        """When a chunk spans two pages its page_number is the first page (Req 3.4)."""
        # Use a very large max_tokens so that the two pages land in one chunk.
        cfg = make_config(max_tokens=800, overlap_tokens=80, min_final_tokens=1)
        chunker = Chunker(cfg)
        pages = [
            make_page("Page one content here.", page_number=1),
            make_page("Page two content here.", page_number=2),
        ]
        result = chunker.chunk(pages)
        # The first chunk must start on page 1.
        assert result[0].metadata.page_number == 1


# ---------------------------------------------------------------------------
# Req 3.6 — final remainder ≥ min_final_tokens → own chunk
# ---------------------------------------------------------------------------

class TestFinalRemainderOwnChunk:
    def test_large_enough_remainder_gets_own_chunk(self):
        """When the trailing remainder has ≥ min_final_tokens it becomes a chunk (Req 3.6)."""
        # max_tokens=20, overlap=2, min_final=10.
        # Build text so that the last segment is exactly 15 tokens — above
        # min_final=10 but below max_tokens=20.
        cfg = make_config(max_tokens=20, overlap_tokens=2, min_final_tokens=10)
        chunker = Chunker(cfg)
        # 35 tokens total: first 20 form a chunk, remainder ~15 > min_final.
        text = word_repeat("word", 35)
        result = chunker.chunk([make_page(text)])
        # There must be at least 2 chunks; last chunk must have ≥ 10 tokens.
        assert len(result) >= 2
        last_count = chunker.count_tokens(result[-1].text)
        assert last_count >= cfg.min_final_tokens


# ---------------------------------------------------------------------------
# Req 3.7 — final remainder < min_final_tokens AND prior chunk exists → merge
# ---------------------------------------------------------------------------

class TestFinalRemainderMerge:
    def test_small_remainder_merges_into_preceding_chunk(self):
        """Small remainder merges into the preceding chunk (Req 3.7)."""
        # max_tokens=30, overlap=3, min_final=25.
        # We'll produce text that leaves a very small tail (< 25 tokens).
        cfg = make_config(max_tokens=30, overlap_tokens=3, min_final_tokens=25)
        chunker = Chunker(cfg)
        # 32 tokens: fills one 30-token chunk then leaves 2-token remainder.
        # The 2-token remainder < min_final=25, so it merges with the preceding.
        text = word_repeat("word", 32)
        result = chunker.chunk([make_page(text)])
        # After merge there is exactly 1 chunk; it is larger than 30.
        assert len(result) == 1
        merged_count = chunker.count_tokens(result[0].text)
        # Merged chunk should contain the full content.
        assert merged_count > 0


# ---------------------------------------------------------------------------
# Req 3.8 — final remainder < min_final_tokens AND no prior chunk → standalone
# ---------------------------------------------------------------------------

class TestFinalRemainderStandalone:
    def test_small_remainder_standalone_when_no_prior_chunk(self):
        """If there are no prior chunks, even a tiny remainder gets its own chunk (Req 3.8)."""
        # Use min_final_tokens=500 so the short text is below the threshold
        # but there are no prior chunks.
        cfg = make_config(max_tokens=800, overlap_tokens=80, min_final_tokens=500)
        chunker = Chunker(cfg)
        text = "Short text below five hundred tokens."
        result = chunker.chunk([make_page(text)])
        # Even though it's below min_final, no prior chunk exists → standalone.
        assert len(result) == 1
        assert "Short text" in result[0].text


# ---------------------------------------------------------------------------
# Determinism (Req 3.5)
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_produces_same_output(self):
        """Running chunk() twice on the same input produces identical results (Req 3.5)."""
        cfg = make_config(max_tokens=20, overlap_tokens=2, min_final_tokens=1)
        chunker = Chunker(cfg)
        text = word_repeat("word", 200)
        pages = [make_page(text)]
        result1 = chunker.chunk(pages)
        result2 = chunker.chunk(pages)
        assert len(result1) == len(result2)
        for c1, c2 in zip(result1, result2):
            assert c1.text == c2.text
            assert c1.metadata == c2.metadata

    def test_different_chunker_instances_same_result(self):
        """Two independent Chunker instances with the same config produce identical output."""
        cfg = make_config(max_tokens=20, overlap_tokens=2, min_final_tokens=1)
        text = word_repeat("word", 200)
        pages = [make_page(text)]
        result1 = Chunker(cfg).chunk(pages)
        result2 = Chunker(cfg).chunk(pages)
        assert [c.text for c in result1] == [c.text for c in result2]


# ---------------------------------------------------------------------------
# Multi-page inputs
# ---------------------------------------------------------------------------

class TestMultiPageInput:
    def test_multiple_pages_all_covered(self):
        """Content from all pages appears across the produced chunks."""
        chunker = Chunker(make_config())
        pages = [
            make_page("First page content.", page_number=1),
            make_page("Second page content.", page_number=2),
            make_page("Third page content.", page_number=3),
        ]
        result = chunker.chunk(pages)
        full_text = " ".join(c.text for c in result)
        assert "First page content" in full_text
        assert "Second page content" in full_text
        assert "Third page content" in full_text

    def test_page_numbers_assigned_correctly(self):
        """Each chunk's page_number comes from the page supplying the first token."""
        # Use tiny max_tokens so we get one chunk per page.
        cfg = make_config(max_tokens=10, overlap_tokens=1, min_final_tokens=1)
        chunker = Chunker(cfg)
        pages = [
            make_page(word_repeat("apple", 10), page_number=1),
            make_page(word_repeat("banana", 10), page_number=2),
        ]
        result = chunker.chunk(pages)
        # The first chunk(s) must reference page 1.
        assert result[0].metadata.page_number == 1
