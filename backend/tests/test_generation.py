"""
backend/tests/test_generation.py

Unit tests for backend/generation.py — AnswerGenerator

Tests cover:
- build_citations: one Citation per distinct filename with sorted unique pages (Req 8.4)
- build_prompt: CoT structure, history inclusion, language instruction, image marker (Req 8.1–8.3, 8.7)
- generate: empty-chunk response (Req 8.5), happy path (Req 8.1), language detection (Req 8.7),
  timeout → error result (Req 8.8, 8.9), provider error → error result (Req 8.8, 8.9)
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import sys
import os

import pytest

from backend.data_models import Candidate, ChunkMetadata, Citation, GenerationResult, Turn
from backend.generation import AnswerGenerator, _NO_INFO_ANSWER, _IMAGE_MARKER


# ---------------------------------------------------------------------------
# Stub LLMProvider
# ---------------------------------------------------------------------------

class StubProvider:
    """Minimal LLMProvider that returns a fixed answer stream."""

    name = "stub"

    def __init__(self, tokens: list[str] | None = None, raise_exc: Exception | None = None):
        self._tokens = tokens or ["Answer text."]
        self._raise_exc = raise_exc
        self.call_count = 0
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
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._token_gen()

    async def _token_gen(self) -> AsyncIterator[str]:
        for token in self._tokens:
            yield token


class TimeoutProvider:
    """LLMProvider that always raises asyncio.TimeoutError."""

    name = "timeout"
    call_count = 0

    async def generate(
        self,
        prompt: str,
        *,
        timeout_s: float,
        stream: bool,
    ) -> AsyncIterator[str]:
        self.call_count += 1
        raise asyncio.TimeoutError("Provider timed out")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_meta(filename: str = "doc.pdf", page: int = 1) -> ChunkMetadata:
    return ChunkMetadata(
        pdf_id="pdf1",
        filename=filename,
        page_number=page,
        chunk_position=0,
        language="en",
    )


def make_candidate(text: str = "chunk text", filename: str = "doc.pdf", page: int = 1) -> Candidate:
    return Candidate(text=text, score=0.9, metadata=make_meta(filename, page))


# ---------------------------------------------------------------------------
# build_citations
# ---------------------------------------------------------------------------

class TestBuildCitations:
    def test_empty_chunks_returns_empty(self):
        gen = AnswerGenerator(StubProvider())
        assert gen.build_citations([]) == []

    def test_single_chunk_single_citation(self):
        gen = AnswerGenerator(StubProvider())
        chunks = [make_candidate(filename="a.pdf", page=3)]
        citations = gen.build_citations(chunks)
        assert len(citations) == 1
        assert citations[0].filename == "a.pdf"
        assert citations[0].pages == [3]

    def test_two_chunks_same_file_distinct_pages(self):
        gen = AnswerGenerator(StubProvider())
        chunks = [
            make_candidate(filename="a.pdf", page=2),
            make_candidate(filename="a.pdf", page=5),
        ]
        citations = gen.build_citations(chunks)
        assert len(citations) == 1
        assert citations[0].pages == [2, 5]  # sorted

    def test_two_chunks_same_file_duplicate_page_deduped(self):
        gen = AnswerGenerator(StubProvider())
        chunks = [
            make_candidate(filename="a.pdf", page=4),
            make_candidate(filename="a.pdf", page=4),
        ]
        citations = gen.build_citations(chunks)
        assert len(citations) == 1
        assert citations[0].pages == [4]

    def test_two_distinct_files_two_citations(self):
        gen = AnswerGenerator(StubProvider())
        chunks = [
            make_candidate(filename="a.pdf", page=1),
            make_candidate(filename="b.pdf", page=7),
        ]
        citations = gen.build_citations(chunks)
        assert len(citations) == 2
        filenames = [c.filename for c in citations]
        assert "a.pdf" in filenames
        assert "b.pdf" in filenames

    def test_pages_sorted_ascending(self):
        gen = AnswerGenerator(StubProvider())
        chunks = [
            make_candidate(filename="x.pdf", page=10),
            make_candidate(filename="x.pdf", page=2),
            make_candidate(filename="x.pdf", page=7),
        ]
        citations = gen.build_citations(chunks)
        assert citations[0].pages == [2, 7, 10]

    def test_citation_order_by_first_occurrence(self):
        gen = AnswerGenerator(StubProvider())
        chunks = [
            make_candidate(filename="z.pdf", page=1),
            make_candidate(filename="a.pdf", page=1),
            make_candidate(filename="z.pdf", page=2),
        ]
        citations = gen.build_citations(chunks)
        # z.pdf appears first in chunks, so it should be first in citations
        assert citations[0].filename == "z.pdf"
        assert citations[1].filename == "a.pdf"


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_prompt_contains_query(self):
        gen = AnswerGenerator(StubProvider())
        chunks = [make_candidate()]
        prompt = gen.build_prompt("What is X?", chunks, [], "en")
        assert "What is X?" in prompt

    def test_prompt_contains_chunk_text(self):
        gen = AnswerGenerator(StubProvider())
        chunks = [make_candidate(text="The answer is 42.")]
        prompt = gen.build_prompt("query", chunks, [], "en")
        assert "The answer is 42." in prompt

    def test_prompt_contains_filename_page_tag(self):
        gen = AnswerGenerator(StubProvider())
        chunks = [make_candidate(filename="report.pdf", page=7)]
        prompt = gen.build_prompt("query", chunks, [], "en")
        assert "[report.pdf, page 7]" in prompt

    def test_prompt_without_history_has_no_history_section(self):
        gen = AnswerGenerator(StubProvider())
        chunks = [make_candidate()]
        prompt = gen.build_prompt("query", chunks, [], "en")
        assert "Conversation History" not in prompt

    def test_prompt_with_history_contains_turns(self):
        gen = AnswerGenerator(StubProvider())
        chunks = [make_candidate()]
        history = [Turn(question="Previous Q", answer="Previous A")]
        prompt = gen.build_prompt("query", chunks, history, "en")
        assert "Previous Q" in prompt
        assert "Previous A" in prompt

    def test_english_prompt_no_language_instruction(self):
        gen = AnswerGenerator(StubProvider())
        chunks = [make_candidate()]
        prompt = gen.build_prompt("query", chunks, [], "en")
        # No explicit language override when query is English
        assert "IMPORTANT" not in prompt or "en" not in prompt.split("IMPORTANT")[1][:50]

    def test_non_english_prompt_includes_language_instruction(self):
        gen = AnswerGenerator(StubProvider())
        chunks = [make_candidate()]
        prompt = gen.build_prompt("Quelle est la réponse?", chunks, [], "fr")
        assert "fr" in prompt
        assert "IMPORTANT" in prompt

    def test_und_language_no_language_instruction(self):
        gen = AnswerGenerator(StubProvider())
        chunks = [make_candidate()]
        prompt = gen.build_prompt("query", chunks, [], "und")
        assert "IMPORTANT" not in prompt

    def test_image_marker_instruction_when_present(self):
        gen = AnswerGenerator(StubProvider())
        text_with_image = f"Some text\n{_IMAGE_MARKER} bar chart showing sales"
        chunks = [make_candidate(text=text_with_image)]
        prompt = gen.build_prompt("query", chunks, [], "en")
        assert "diagram" in prompt.lower() or "image" in prompt.lower()

    def test_no_image_marker_instruction_when_absent(self):
        gen = AnswerGenerator(StubProvider())
        chunks = [make_candidate(text="Plain text without any image marker.")]
        prompt = gen.build_prompt("query", chunks, [], "en")
        assert _IMAGE_MARKER not in prompt

    def test_cot_instruction_present(self):
        gen = AnswerGenerator(StubProvider())
        chunks = [make_candidate()]
        prompt = gen.build_prompt("query", chunks, [], "en")
        assert "step" in prompt.lower()

    def test_multiple_chunks_all_tagged(self):
        gen = AnswerGenerator(StubProvider())
        chunks = [
            make_candidate(filename="a.pdf", page=1),
            make_candidate(filename="b.pdf", page=5),
        ]
        prompt = gen.build_prompt("query", chunks, [], "en")
        assert "[a.pdf, page 1]" in prompt
        assert "[b.pdf, page 5]" in prompt


# ---------------------------------------------------------------------------
# generate — async
# ---------------------------------------------------------------------------

class TestGenerate:
    @pytest.mark.asyncio
    async def test_empty_chunks_returns_no_info(self):
        """Req 8.5: empty chunks → no-info response, zero citations."""
        gen = AnswerGenerator(StubProvider())
        result = await gen.generate("query", [], [])
        assert result.answer == _NO_INFO_ANSWER
        assert result.citations == []
        assert result.error is None

    @pytest.mark.asyncio
    async def test_empty_chunks_no_provider_call(self):
        """Req 8.5: provider must NOT be called for empty chunks."""
        provider = StubProvider()
        gen = AnswerGenerator(provider)
        await gen.generate("query", [], [])
        assert provider.call_count == 0

    @pytest.mark.asyncio
    async def test_happy_path_returns_answer(self):
        """Req 8.1: non-empty chunks → call provider, return answer."""
        provider = StubProvider(tokens=["The answer is 42."])
        gen = AnswerGenerator(provider)
        chunks = [make_candidate()]
        result = await gen.generate("What is the answer?", chunks, [])
        assert result.answer == "The answer is 42."
        assert result.error is None

    @pytest.mark.asyncio
    async def test_happy_path_builds_citations(self):
        """Req 8.4: citations are built from chunks."""
        provider = StubProvider()
        gen = AnswerGenerator(provider)
        chunks = [
            make_candidate(filename="a.pdf", page=1),
            make_candidate(filename="b.pdf", page=3),
        ]
        result = await gen.generate("query", chunks, [])
        assert len(result.citations) == 2
        filenames = {c.filename for c in result.citations}
        assert filenames == {"a.pdf", "b.pdf"}

    @pytest.mark.asyncio
    async def test_provider_called_exactly_once(self):
        """Req 8.2: single generation pass, no multi-step."""
        provider = StubProvider()
        gen = AnswerGenerator(provider)
        chunks = [make_candidate()]
        await gen.generate("query", chunks, [])
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_timeout_returns_error_result(self):
        """Req 8.8, 8.9: timeout → error result, empty answer, zero citations."""
        provider = TimeoutProvider()
        gen = AnswerGenerator(provider, generation_timeout_s=0.01)
        chunks = [make_candidate()]
        result = await gen.generate("query", chunks, [])
        assert result.answer == ""
        assert result.citations == []
        assert result.error is not None
        assert len(result.error) > 0

    @pytest.mark.asyncio
    async def test_provider_runtime_error_returns_error_result(self):
        """Req 8.8, 8.9: provider exception → error result, empty answer, zero citations."""
        provider = StubProvider(raise_exc=RuntimeError("API failure"))
        gen = AnswerGenerator(provider)
        chunks = [make_candidate()]
        result = await gen.generate("query", chunks, [])
        assert result.answer == ""
        assert result.citations == []
        assert result.error is not None
        assert "API failure" in result.error or "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_partial_answer_on_error(self):
        """Req 8.9: error responses must not include any partial answer text."""
        provider = StubProvider(raise_exc=RuntimeError("oops"))
        gen = AnswerGenerator(provider)
        chunks = [make_candidate()]
        result = await gen.generate("query", chunks, [])
        assert result.answer == ""

    @pytest.mark.asyncio
    async def test_language_detection_non_english(self):
        """Req 8.7: non-English query → prompt includes language instruction."""
        provider = StubProvider()
        gen = AnswerGenerator(provider)
        # Use a clearly French query (long enough to detect language)
        query = "Quelle est la capitale de la France et pourquoi est-elle importante?"
        chunks = [make_candidate()]
        await gen.generate(query, chunks, [])
        assert provider.last_prompt is not None
        # The prompt should contain the detected language code
        # (fr is detected from the French query text)
        assert "fr" in provider.last_prompt or "IMPORTANT" in provider.last_prompt

    @pytest.mark.asyncio
    async def test_history_is_included_in_prompt(self):
        """Req 12.2: prior turns are injected into the prompt."""
        provider = StubProvider()
        gen = AnswerGenerator(provider)
        history = [Turn(question="Earlier Q", answer="Earlier A")]
        chunks = [make_candidate()]
        await gen.generate("follow-up question", chunks, history)
        assert provider.last_prompt is not None
        assert "Earlier Q" in provider.last_prompt
        assert "Earlier A" in provider.last_prompt

    @pytest.mark.asyncio
    async def test_multiple_tokens_concatenated(self):
        """Verify that multiple streamed tokens are joined into a single answer."""
        provider = StubProvider(tokens=["Hello ", "world", "!"])
        gen = AnswerGenerator(provider)
        chunks = [make_candidate()]
        result = await gen.generate("query", chunks, [])
        assert result.answer == "Hello world!"

    @pytest.mark.asyncio
    async def test_citations_sorted_unique_pages(self):
        """Citations aggregated from chunks have sorted unique page numbers."""
        provider = StubProvider()
        gen = AnswerGenerator(provider)
        chunks = [
            make_candidate(filename="doc.pdf", page=5),
            make_candidate(filename="doc.pdf", page=2),
            make_candidate(filename="doc.pdf", page=5),   # duplicate page
        ]
        result = await gen.generate("query", chunks, [])
        assert len(result.citations) == 1
        assert result.citations[0].pages == [2, 5]

    @pytest.mark.asyncio
    async def test_error_result_has_zero_citations(self):
        """Req 8.9: error result must have zero citations."""
        provider = StubProvider(raise_exc=ValueError("broken"))
        gen = AnswerGenerator(provider)
        chunks = [
            make_candidate(filename="a.pdf", page=1),
            make_candidate(filename="b.pdf", page=2),
        ]
        result = await gen.generate("query", chunks, [])
        assert result.citations == []
