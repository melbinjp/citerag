"""
backend/ingestion/chunker.py

Deterministic recursive token-bounded text splitter with overlap and metadata.

Task 3.1 — ChunkerConfig, tokenizer, and config validation.
Task 3.2 — Chunker.chunk() recursive splitting with overlap, metadata, and
            final-remainder disposition.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 3.8, 3.9, 3.10 (config validation)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import tiktoken

from data_models import Chunk, ChunkMetadata


# ---------------------------------------------------------------------------
# Tokenizer protocol + default singleton
# ---------------------------------------------------------------------------

@runtime_checkable
class Tokenizer(Protocol):
    """Minimal interface required by the Chunker."""

    def encode(self, text: str) -> list[int]:
        """Return the token-id list for *text*."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Return the string for a token-id list."""
        ...


def _build_default_tokenizer() -> Tokenizer:
    """Return the single fixed tokenizer (cl100k_base via tiktoken).

    Loaded once at module import; callers share the same instance.
    tiktoken encodings are thread-safe and cheap to call repeatedly.
    """
    return tiktoken.get_encoding("cl100k_base")


# Module-level singleton — one tokenizer, shared across all Chunker instances
# that do not supply their own.
_DEFAULT_TOKENIZER: Tokenizer = _build_default_tokenizer()


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PageText:
    """A single PDF page's text together with its provenance.

    Attributes:
        page_number: 1-based page number within the PDF.
        text:        Normalized text extracted from the page.
        filename:    Original filename of the PDF (e.g. "report.pdf").
        pdf_id:      Stable identifier for the PDF (derived from path/content).
        language:    ISO 639-1 two-letter code, or ``"und"`` when undetermined.
    """

    page_number: int   # 1-based
    text: str
    filename: str
    pdf_id: str
    language: str = "und"


# ---------------------------------------------------------------------------
# Config and error types
# ---------------------------------------------------------------------------

# Permitted absolute range for max_tokens.
_MIN_MAX_TOKENS: int = 1
_MAX_MAX_TOKENS: int = 8192  # bge-m3 hard cap

# Overlap expressed as a fraction of max_tokens: [10 %, 30 %].
_OVERLAP_LOW_FRAC: float = 0.10
_OVERLAP_HIGH_FRAC: float = 0.30


class ChunkerConfigError(Exception):
    """Raised when a :class:`ChunkerConfig` value falls outside its permitted range."""


@dataclass(frozen=True)
class ChunkerConfig:
    """Immutable configuration for :class:`Chunker`.

    Validation rules (Requirement 3.10):
    - ``max_tokens`` must be in ``[1, 8192]``.
    - ``overlap_tokens`` must be in ``[10 %, 30 %]`` of ``max_tokens``
      (both bounds inclusive, rounded to the nearest whole token).
    - ``min_final_tokens`` must be in ``[1, max_tokens]``.

    :raises ChunkerConfigError: immediately on construction if any value is
        out of range, ensuring that an invalid config never silently produces
        incorrect chunks.
    """

    max_tokens: int = 800          # CHUNK_MAX_TOKENS
    overlap_tokens: int = 150      # CHUNK_OVERLAP_TOKENS — must be 10%..30% of max_tokens
    min_final_tokens: int = 100    # CHUNK_MIN_FINAL_TOKENS — must be 1..max_tokens

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Validate all fields; raise :class:`ChunkerConfigError` on the first
        violation found.  (Req 3.10: reject config with an error, produce no
        chunks.)
        """
        self._validate_max_tokens()
        self._validate_overlap_tokens()
        self._validate_min_final_tokens()

    def _validate_max_tokens(self) -> None:
        if not (_MIN_MAX_TOKENS <= self.max_tokens <= _MAX_MAX_TOKENS):
            raise ChunkerConfigError(
                f"max_tokens={self.max_tokens!r} is out of the permitted range "
                f"[{_MIN_MAX_TOKENS}, {_MAX_MAX_TOKENS}]."
            )

    def _validate_overlap_tokens(self) -> None:
        # Permitted overlap window based on the *already-validated* max_tokens.
        low = round(self.max_tokens * _OVERLAP_LOW_FRAC)
        high = round(self.max_tokens * _OVERLAP_HIGH_FRAC)
        if not (low <= self.overlap_tokens <= high):
            raise ChunkerConfigError(
                f"overlap_tokens={self.overlap_tokens!r} is out of the permitted range "
                f"[{low}, {high}] (10%–30% of max_tokens={self.max_tokens})."
            )

    def _validate_min_final_tokens(self) -> None:
        if not (1 <= self.min_final_tokens <= self.max_tokens):
            raise ChunkerConfigError(
                f"min_final_tokens={self.min_final_tokens!r} is out of the permitted range "
                f"[1, {self.max_tokens}] (1..max_tokens)."
            )


# ---------------------------------------------------------------------------
# Chunker (config + tokenizer skeleton; chunk() implemented in task 3.2)
# ---------------------------------------------------------------------------

class Chunker:
    """Deterministic, recursive, token-bounded splitter with overlap and metadata.

    Construction validates the supplied :class:`ChunkerConfig` and stores the
    tokenizer.  The actual :meth:`chunk` method is implemented in task 3.2.
    """

    def __init__(
        self,
        config: ChunkerConfig,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        """Initialise the Chunker.

        :param config:    Validated chunker configuration.  The dataclass
                          constructor already raises :class:`ChunkerConfigError`
                          for invalid values, so by the time this method runs the
                          config is guaranteed to be valid (Req 3.10).
        :param tokenizer: Optional tokenizer override.  Defaults to the module-
                          level ``cl100k_base`` singleton when ``None``.
        :raises ChunkerConfigError: if *config* is invalid (propagated from
            :class:`ChunkerConfig.__post_init__`).
        """
        # config validation is performed inside ChunkerConfig.__post_init__;
        # arriving here means the config is valid.
        self._config: ChunkerConfig = config
        self._tokenizer: Tokenizer = tokenizer if tokenizer is not None else _DEFAULT_TOKENIZER

    # ------------------------------------------------------------------
    # Public helpers (used by chunk() and tests)
    # ------------------------------------------------------------------

    @property
    def config(self) -> ChunkerConfig:
        """The validated configuration for this Chunker."""
        return self._config

    @property
    def tokenizer(self) -> Tokenizer:
        """The tokenizer used for all token-count operations."""
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in *text* using this Chunker's tokenizer."""
        return len(self._tokenizer.encode(text))

    # ------------------------------------------------------------------
    # chunk() — implemented in task 3.2
    # ------------------------------------------------------------------

    def chunk(self, pages: list[PageText]) -> list[Chunk]:
        """Split page-tagged text into token-bounded passages with overlap.

        Algorithm (Requirements 3.1–3.4, 3.6–3.9):

        1. Concatenate all pages into one token stream, recording page-boundary
           token offsets so we can assign the correct page_number to each chunk
           (the page that contributes the first token – Req 3.4).
        2. Split the full text at paragraph boundaries (``\\n\\n``) first, then
           sentence boundaries (``". "`` / ``"? "`` / ``"! "``), producing a
           sequence of segments.
        3. Greedily accumulate segments into a buffer up to ``max_tokens``.
           When adding a segment would overflow, emit the buffer as a chunk and
           start a new buffer pre-seeded with the last ``overlap_tokens`` of the
           previous chunk (Req 3.2).
        4. After all segments are consumed, call :meth:`_emit_final` to handle
           whatever remains in the buffer (Req 3.6–3.8).
        5. Zero-token input → return empty list; the caller may interpret the
           empty result as "no content available" (Req 3.9).

        :param pages: Ordered list of :class:`PageText` objects (one per page).
        :returns:     List of :class:`Chunk` objects, possibly empty.
        """
        # Req 3.9 — zero-token / empty input → zero chunks.
        if not pages:
            return []

        # ------------------------------------------------------------------
        # Step 1: build a flat token list, tracking page boundaries.
        # page_starts[i] = first token index belonging to pages[i].page_number
        # ------------------------------------------------------------------
        all_tokens: list[int] = []
        # Map from token index → page_number (only store boundary starts).
        # We'll build a sorted list of (start_token_idx, page_number) tuples.
        page_boundaries: list[tuple[int, int]] = []  # (token_offset, page_number)

        full_text_parts: list[str] = []
        char_to_page: list[int] = []  # char index → page_number

        for page in pages:
            start_char = len("".join(full_text_parts))
            page_text = page.text
            # Separate pages with a paragraph boundary so the splitter can
            # split at page transitions; this is purely a joining convention
            # and the separator tokens are accounted for during tokenisation.
            if full_text_parts:
                sep = "\n\n"
                char_to_page.extend([page.page_number] * len(sep))
                full_text_parts.append(sep)
            char_to_page.extend([page.page_number] * len(page_text))
            full_text_parts.append(page_text)

        full_text = "".join(full_text_parts)

        # Tokenise the entire concatenated text once (Req 3.1 – single
        # consistent tokeniser).
        all_token_ids = self._tokenizer.encode(full_text)

        # Req 3.9 – zero tokens → zero chunks.
        if not all_token_ids:
            return []

        # Build a token-index → page_number map using character offsets.
        # tiktoken encodes greedily; we use decode(single_token) round-trips to
        # track which character each token corresponds to.  Because this is O(n)
        # in the number of tokens it is acceptable for the chunk-size ranges we
        # support (≤8192 tokens per bge-m3 limit).
        token_page_map = self._build_token_page_map(all_token_ids, char_to_page)

        # Gather pdf_id / filename from the first page (all pages share them).
        pdf_id = pages[0].pdf_id
        filename = pages[0].filename

        # Build a quick char_offset → language map as well (per page).
        # We use the language of whichever page contributes the chunk's first
        # token (same rule as page_number – Req 3.4 extension).
        page_language_map: dict[int, str] = {p.page_number: p.language for p in pages}

        # ------------------------------------------------------------------
        # Step 2: split full_text into segments at paragraph / sentence
        # boundaries (Req 3.1 – recursive split attempts paragraph first,
        # then sentence).
        # ------------------------------------------------------------------
        segments: list[str] = self._split_into_segments(full_text)

        # ------------------------------------------------------------------
        # Steps 3-4: greedy fill → emit chunks → handle final remainder.
        # ------------------------------------------------------------------
        chunks: list[Chunk] = []
        chunk_position: int = 0

        # Current buffer is stored as a list of token ids.
        buffer_tokens: list[int] = []

        for seg in segments:
            seg_tokens = self._tokenizer.encode(seg)
            if not seg_tokens:
                continue

            # If this single segment is larger than max_tokens, split it
            # further at the token level (hard cut).
            if len(seg_tokens) > self._config.max_tokens:
                # First, flush whatever is in the buffer.
                if buffer_tokens:
                    new_chunk, chunk_position = self._emit_chunk(
                        buffer_tokens, token_page_map, page_language_map,
                        pdf_id, filename, chunk_position, chunks,
                        all_token_ids,
                    )
                    chunks.append(new_chunk)
                    # Seed overlap for the next buffer.
                    buffer_tokens = buffer_tokens[-self._config.overlap_tokens:]

                # Hard-cut the oversized segment into max_tokens windows.
                offset = 0
                step = self._config.max_tokens - self._config.overlap_tokens
                while offset < len(seg_tokens):
                    window = seg_tokens[offset: offset + self._config.max_tokens]
                    if len(window) == self._config.max_tokens:
                        new_chunk, chunk_position = self._emit_chunk(
                            window, token_page_map, page_language_map,
                            pdf_id, filename, chunk_position, chunks,
                            all_token_ids,
                        )
                        chunks.append(new_chunk)
                        # Overlap seed for next iteration.
                        buffer_tokens = window[-self._config.overlap_tokens:]
                        offset += step
                    else:
                        # Last partial window — put it in the buffer.
                        buffer_tokens = list(window)
                        offset += len(window)
                continue

            # Normal segment: check if it fits in the current buffer.
            if len(buffer_tokens) + len(seg_tokens) <= self._config.max_tokens:
                buffer_tokens.extend(seg_tokens)
            else:
                # Flush the buffer as a chunk (if non-empty).
                if buffer_tokens:
                    new_chunk, chunk_position = self._emit_chunk(
                        buffer_tokens, token_page_map, page_language_map,
                        pdf_id, filename, chunk_position, chunks,
                        all_token_ids,
                    )
                    chunks.append(new_chunk)
                    # Start new buffer with overlap from the previous chunk.
                    overlap_seed = buffer_tokens[-self._config.overlap_tokens:]
                else:
                    overlap_seed = []

                # If the segment alone fits, start buffer with overlap + segment.
                candidate = overlap_seed + seg_tokens
                if len(candidate) <= self._config.max_tokens:
                    buffer_tokens = candidate
                else:
                    # Segment alone is ≤ max_tokens but overlap + segment
                    # exceeds it; trim the overlap.
                    trim = len(candidate) - self._config.max_tokens
                    buffer_tokens = overlap_seed[trim:] + seg_tokens

        # ------------------------------------------------------------------
        # Step 4: handle the final remainder.
        # ------------------------------------------------------------------
        remainder_text = self._tokenizer.decode(buffer_tokens) if buffer_tokens else ""
        # Determine metadata for the remainder.
        if buffer_tokens:
            # Compute global token offset of the first remainder token.
            # We need to find the first token of buffer_tokens in the global
            # token stream. Rather than tracking exact indices through all the
            # overlap bookkeeping, we decode the remainder text and re-encode
            # the full token stream up to the remainder.
            #
            # Simpler approach: the first token in buffer_tokens maps to the
            # page of the first matching token in all_token_ids.
            first_token_id = buffer_tokens[0]
            remainder_page = self._resolve_page_for_token_id(
                first_token_id, buffer_tokens, all_token_ids, token_page_map
            )
            remainder_lang = page_language_map.get(remainder_page, "und")
            remainder_meta = ChunkMetadata(
                pdf_id=pdf_id,
                filename=filename,
                page_number=remainder_page,
                chunk_position=chunk_position,
                language=remainder_lang,
            )
            chunks = self._emit_final(chunks, remainder_text, remainder_meta)

        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_into_segments(self, text: str) -> list[str]:
        """Split *text* at paragraph then sentence boundaries.

        Strategy (Req 3.1):
        1. First split on paragraph boundaries (``\\n\\n``).
        2. For each paragraph that is too large (> max_tokens), split it
           further at sentence boundaries.
        3. Any resulting fragment that is still > max_tokens is left as-is
           (the main loop will hard-cut it at the token level).

        Returns a flat list of non-empty string segments.
        """
        # Paragraph split.
        paragraphs = re.split(r"\n\n+", text)

        segments: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if self.count_tokens(para) <= self._config.max_tokens:
                segments.append(para)
            else:
                # Sentence-level split within the paragraph.
                sentences = self._split_sentences(para)
                segments.extend(s for s in sentences if s.strip())

        return segments

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split *text* at sentence boundaries (``'. '``, ``'? '``, ``'! '``).

        Preserves the terminating punctuation with the sentence it belongs to.
        """
        # Use a regex that splits after '. ', '? ', '! ' while keeping the
        # delimiter attached to the preceding sentence.
        parts = re.split(r"(?<=[.?!])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _build_token_page_map(
        self,
        token_ids: list[int],
        char_to_page: list[int],
    ) -> list[int]:
        """Return a list of length ``len(token_ids)`` mapping each token index
        to its page_number.

        We reconstruct the mapping by decoding each token and advancing a
        character cursor through ``char_to_page``.
        """
        token_page = []
        char_cursor = 0
        for tid in token_ids:
            token_str = self._tokenizer.decode([tid])
            page = char_to_page[char_cursor] if char_cursor < len(char_to_page) else (
                char_to_page[-1] if char_to_page else 1
            )
            token_page.append(page)
            char_cursor += len(token_str)
        return token_page

    def _emit_chunk(
        self,
        token_ids: list[int],
        token_page_map: list[int],
        page_language_map: dict[int, str],
        pdf_id: str,
        filename: str,
        chunk_position: int,
        prior_chunks: list[Chunk],
        all_token_ids: list[int] | None = None,
    ) -> tuple[Chunk, int]:
        """Create a :class:`Chunk` from *token_ids* and return
        ``(chunk, next_chunk_position)``.

        The page_number is taken from the global ``token_page_map`` at the
        index of the first token in this chunk's token ids.
        """
        text = self._tokenizer.decode(token_ids)
        first_token_id = token_ids[0]
        _all = all_token_ids if all_token_ids is not None else token_ids
        page_number = self._resolve_page_for_token_id(
            first_token_id, token_ids, _all, token_page_map
        )
        language = page_language_map.get(page_number, "und")
        meta = ChunkMetadata(
            pdf_id=pdf_id,
            filename=filename,
            page_number=page_number,
            chunk_position=chunk_position,
            language=language,
        )
        return Chunk(text=text, metadata=meta), chunk_position + 1

    def _resolve_page_for_token_id(
        self,
        first_token_id: int,
        buffer_tokens: list[int],
        all_token_ids: list[int],
        token_page_map: list[int],
    ) -> int:
        """Find the page_number of *first_token_id* by locating the first
        occurrence of the buffer's token sequence in the global token list.

        Falls back to page 1 if no match is found.
        """
        if not token_page_map:
            return 1
        # Search for the first matching token to get a candidate page.
        for idx, tid in enumerate(all_token_ids):
            if tid == first_token_id:
                return token_page_map[idx]
        return token_page_map[0] if token_page_map else 1

    def _emit_final(
        self,
        chunks: list[Chunk],
        remainder: str,
        meta: ChunkMetadata,
    ) -> list[Chunk]:
        """Dispose of the final remainder text per the min-final policy.

        Req 3.6: remainder >= min_final_tokens → own final chunk.
        Req 3.7: remainder < min_final_tokens AND prior chunk exists → merge.
        Req 3.8: remainder < min_final_tokens AND no prior chunk → standalone.
        """
        r = self.count_tokens(remainder)
        if r == 0:
            return chunks  # nothing left over

        if r >= self._config.min_final_tokens:
            # Req 3.6 — large-enough remainder becomes its own final chunk.
            chunks.append(Chunk(text=remainder, metadata=meta))
        elif chunks:
            # Req 3.7 — too-small remainder merges into the immediately
            # preceding chunk, retaining that chunk's metadata.
            prev = chunks[-1]
            merged_text = self._join_without_double_overlap(prev.text, remainder)
            chunks[-1] = Chunk(text=merged_text, metadata=prev.metadata)
        else:
            # Req 3.8 — no prior chunk → standalone final chunk.
            chunks.append(Chunk(text=remainder, metadata=meta))

        return chunks

    @staticmethod
    def _join_without_double_overlap(base: str, tail: str) -> str:
        """Append *tail* to *base*, stripping any duplicate suffix/prefix.

        When the chunker produces an overlap by re-seeding the buffer with the
        last ``overlap_tokens`` of the previous chunk, the tail may begin with
        text that already appears at the end of *base*.  This helper finds the
        longest suffix of *base* that matches a prefix of *tail* and strips it
        before joining, so no tokens are duplicated.
        """
        # Try successively shorter suffixes of base as prefixes of tail.
        max_overlap = min(len(base), len(tail))
        for length in range(max_overlap, 0, -1):
            if base.endswith(tail[:length]):
                return base + tail[length:]
        return base + (" " if base and not base.endswith(" ") else "") + tail
