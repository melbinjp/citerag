"""
backend/generation.py

AnswerGenerator — single-pass Chain-of-Thought answer generation with
structured provenance citations.

Design responsibilities
-----------------------
* Build a single CoT prompt that injects the Top_K retrieved chunks, the
  conversation history, and inline ``[filename, page X]`` attribution
  instructions (Req 8.1, 8.2).
* Reference Semantic_Markdown_Wrapping image markers contextually (Req 8.3).
* Append one structured ``Citation`` per distinct source filename with sorted
  unique page numbers (Req 8.4).
* Return "no relevant information found" + zero citations when chunks are
  empty (Req 8.5).
* Detect query language; answer in the same language when it is not English
  (Req 8.7).
* On any provider error or timeout, return an error result with no partial
  answer and zero citations (Req 8.8, 8.9).

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.7, 8.8, 8.9
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from backend.data_models import Candidate, Citation, GenerationResult, Turn
from backend.ingestion.language import LanguageDetector

if TYPE_CHECKING:
    from backend.providers import LLMProvider

logger = logging.getLogger(__name__)

# Sentinel marker used by Semantic_Markdown_Wrapping (Req 1.3, 1.4, 8.3)
_IMAGE_MARKER = "> [System Note: Image Content]"

# The "no relevant information" answer text (Req 8.5)
_NO_INFO_ANSWER = "I could not find relevant information to answer your question."

# Default timeout for provider calls, in seconds (Req 8.8)
_DEFAULT_TIMEOUT_S: float = 30.0


class AnswerGenerator:
    """Synthesizes an answer from retrieved chunks using a single CoT prompt.

    Parameters
    ----------
    provider:
        An ``LLMProvider`` instance (e.g. ``GeminiProvider``) that will
        receive the assembled prompt and stream/yield answer tokens.
    generation_timeout_s:
        Hard deadline in seconds for a single generation call.  Defaults to
        30 s; must be in the range 1–120 s as per ``GENERATION_TIMEOUT_S``
        configuration (Req 8.8).
    """

    def __init__(
        self,
        provider: "LLMProvider",
        generation_timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._provider = provider
        self._timeout_s = generation_timeout_s
        self._language_detector = LanguageDetector()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        query: str,
        chunks: list[Candidate],
        history: list[Turn],
        answer_language: str,
    ) -> str:
        """Assemble the single-pass CoT system prompt.

        The prompt:
        * Summarises the role (answer only from the provided passages).
        * Includes conversation history when present (Req 12.2).
        * Lists each chunk prefixed with its ``[filename, page X]`` tag so
          the model can cite it inline (Req 8.4).
        * Notes any image-content markers so the model can reference them
          contextually (Req 8.3).
        * If ``answer_language != "en"``, instructs the model to respond in
          that language (Req 8.7).
        * Instructs the model to append ``[filename, page X]`` inline after
          every claim derived from a chunk (Req 8.4).

        Parameters
        ----------
        query:
            The user's question.
        chunks:
            Retrieved ``Candidate`` objects — must be non-empty (callers
            should not reach this method with an empty list).
        history:
            Prior conversation turns, oldest first.
        answer_language:
            ISO 639-1 code of the query's detected language, e.g. ``"en"``,
            ``"fr"``.  ``"und"`` is treated as English.

        Returns
        -------
        str
            A fully assembled prompt string ready for the LLM provider.
        """
        lines: list[str] = []

        # ── Role instruction ──────────────────────────────────────────
        lines.append(
            "You are a precise, fact-grounded question-answering assistant. "
            "Answer the user's question using ONLY the information provided "
            "in the passages below. Do not incorporate any knowledge not "
            "contained in those passages."
        )

        # ── Language instruction (Req 8.7) ────────────────────────────
        if answer_language not in ("en", "und", ""):
            lines.append(
                f"\nIMPORTANT: The user's question is written in language "
                f"'{answer_language}'. You MUST write your entire answer in "
                f"that same language ('{answer_language}')."
            )

        # ── Citation instruction (Req 8.4) ────────────────────────────
        lines.append(
            "\nAfter every claim you make that is derived from a passage, "
            "append the inline provenance tag exactly as shown: "
            "[filename, page X]  "
            "(replace 'filename' with the source filename and 'X' with the "
            "page number).  Do not omit the provenance tag for any claim."
        )

        # ── Image marker instruction (Req 8.3) ────────────────────────
        has_image = any(_IMAGE_MARKER in c.text for c in chunks)
        if has_image:
            lines.append(
                "\nSome passages contain the marker "
                f"'{_IMAGE_MARKER}' followed by text extracted from a "
                "diagram, table, or image in the source document. "
                "When such a passage is relevant, reference the corresponding "
                "diagram or table contextually in your answer."
            )

        # ── Chain-of-Thought reasoning instruction ────────────────────
        lines.append(
            "\nThink step by step before writing your final answer. "
            "Show your reasoning briefly, then give the final answer."
        )

        # ── Conversation history (Req 12.2) ───────────────────────────
        if history:
            lines.append("\n--- Conversation History ---")
            for turn in history:
                lines.append(f"User: {turn.question}")
                lines.append(f"Assistant: {turn.answer}")
            lines.append("--- End of Conversation History ---")

        # ── Retrieved passages (Req 8.1) ──────────────────────────────
        lines.append("\n--- Retrieved Passages ---")
        for idx, candidate in enumerate(chunks, start=1):
            filename = candidate.metadata.filename
            page = candidate.metadata.page_number
            tag = f"[{filename}, page {page}]"
            lines.append(f"\nPassage {idx} {tag}:\n{candidate.text}")
        lines.append("--- End of Retrieved Passages ---")

        # ── User question ─────────────────────────────────────────────
        lines.append(f"\nUser question: {query}")
        lines.append(
            "\nNow reason through the passages and provide your answer, "
            "citing inline provenance tags after every claim."
        )

        return "\n".join(lines)

    def build_citations(self, chunks: list[Candidate]) -> list[Citation]:
        """Build one ``Citation`` per distinct source filename.

        For each unique ``filename`` in *chunks*, collects all page numbers
        from the associated ``Candidate`` objects, de-duplicates them, sorts
        them in ascending order, and returns a ``Citation(filename, pages)``.

        The returned list is ordered by first occurrence of each filename in
        the *chunks* list, which preserves the retrieval ranking order.

        Parameters
        ----------
        chunks:
            Retrieved candidates (may be empty — returns ``[]``).

        Returns
        -------
        list[Citation]
            One entry per distinct filename, pages sorted and de-duplicated.
        """
        # Preserve first-occurrence order via an ordered mapping
        pages_by_file: dict[str, set[int]] = defaultdict(set)
        file_order: list[str] = []

        for candidate in chunks:
            filename = candidate.metadata.filename
            if filename not in pages_by_file:
                file_order.append(filename)
            pages_by_file[filename].add(candidate.metadata.page_number)

        return [
            Citation(filename=fname, pages=sorted(pages_by_file[fname]))
            for fname in file_order
        ]

    async def generate(
        self,
        query: str,
        chunks: list[Candidate],
        history: list[Turn],
    ) -> GenerationResult:
        """Generate an answer for *query* given the retrieved *chunks*.

        Behaviour
        ---------
        * Empty *chunks* → return ``GenerationResult`` with the
          "no relevant information" message and zero citations (Req 8.5).
        * Non-empty *chunks* → detect query language, build the CoT prompt,
          stream tokens from the configured provider, and aggregate the full
          answer text; then return the answer with structured citations
          (Req 8.1, 8.4, 8.7).
        * On any ``Exception`` (including ``asyncio.TimeoutError``) → return
          an error ``GenerationResult`` with an empty answer string, zero
          citations, and a descriptive ``error`` field (Req 8.8, 8.9).

        Parameters
        ----------
        query:
            The user's question text.
        chunks:
            Retrieved candidates from the Retrieval_Service (possibly empty).
        history:
            Ordered prior turns for this Conversation_Session.

        Returns
        -------
        GenerationResult
            ``answer`` — the generated text (or "no relevant info" message).
            ``citations`` — list of ``Citation`` objects (empty on error / no
                           chunks).
            ``error`` — ``None`` on success; a description string on failure.
        """
        # ── Req 8.5: empty chunks → no-info response ──────────────────
        if not chunks:
            return GenerationResult(
                answer=_NO_INFO_ANSWER,
                citations=[],
                error=None,
            )

        # ── Req 8.7: detect query language ────────────────────────────
        answer_language = self._language_detector.detect(query)

        # ── Build prompt (Req 8.1, 8.2) ───────────────────────────────
        prompt = self.build_prompt(query, chunks, history, answer_language)

        # ── Call provider with timeout (Req 8.8) ──────────────────────
        try:
            tokens: list[str] = []
            async_gen = await self._provider.generate(
                prompt,
                timeout_s=self._timeout_s,
                stream=True,
            )
            async for token in async_gen:
                tokens.append(token)

            answer = "".join(tokens).strip()

        except asyncio.TimeoutError as exc:
            # Req 8.8, 8.9: timeout → error result, zero citations, no partial answer
            logger.error(
                "AnswerGenerator: provider timed out after %s s: %s",
                self._timeout_s,
                exc,
            )
            return GenerationResult(
                answer="",
                citations=[],
                error=(
                    f"Answer generation timed out after {self._timeout_s} seconds."
                ),
            )
        except Exception as exc:  # noqa: BLE001
            # Req 8.8, 8.9: provider error → error result, zero citations, no partial answer
            logger.error("AnswerGenerator: provider error: %s", exc)
            return GenerationResult(
                answer="",
                citations=[],
                error=f"Answer generation failed: {exc}",
            )

        # ── Build structured citations (Req 8.4) ──────────────────────
        citations = self.build_citations(chunks)

        return GenerationResult(
            answer=answer,
            citations=citations,
            error=None,
        )
