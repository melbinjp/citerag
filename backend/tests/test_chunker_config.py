"""
tests/test_chunker_config.py

Unit tests for ChunkerConfig validation, the default tokenizer, and the
PageText datatype (task 3.1).

Requirements: 3.10
"""

import pytest

from ingestion.chunker import (
    Chunker,
    ChunkerConfig,
    ChunkerConfigError,
    PageText,
    _DEFAULT_TOKENIZER,
)


# ---------------------------------------------------------------------------
# PageText
# ---------------------------------------------------------------------------

class TestPageText:
    def test_default_language(self):
        pt = PageText(page_number=1, text="hello", filename="a.pdf", pdf_id="abc")
        assert pt.language == "und"

    def test_explicit_language(self):
        pt = PageText(page_number=2, text="bonjour", filename="b.pdf", pdf_id="xyz", language="fr")
        assert pt.language == "fr"

    def test_frozen(self):
        pt = PageText(page_number=1, text="hi", filename="f.pdf", pdf_id="id1")
        with pytest.raises((AttributeError, TypeError)):
            pt.text = "new"  # type: ignore[misc]

    def test_page_number_stored(self):
        pt = PageText(page_number=42, text="", filename="f.pdf", pdf_id="id")
        assert pt.page_number == 42


# ---------------------------------------------------------------------------
# ChunkerConfig — valid defaults
# ---------------------------------------------------------------------------

class TestChunkerConfigDefaults:
    def test_default_values(self):
        cfg = ChunkerConfig()
        assert cfg.max_tokens == 800
        assert cfg.overlap_tokens == 150
        assert cfg.min_final_tokens == 100

    def test_default_is_valid(self):
        # Must not raise
        cfg = ChunkerConfig()
        assert cfg is not None

    def test_frozen(self):
        cfg = ChunkerConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.max_tokens = 100  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ChunkerConfig — valid custom values
# ---------------------------------------------------------------------------

class TestChunkerConfigValid:
    def test_tight_overlap_at_10_percent(self):
        # overlap_tokens == round(400 * 0.10) == 40
        cfg = ChunkerConfig(max_tokens=400, overlap_tokens=40, min_final_tokens=1)
        assert cfg.overlap_tokens == 40

    def test_tight_overlap_at_30_percent(self):
        # overlap_tokens == round(400 * 0.30) == 120
        cfg = ChunkerConfig(max_tokens=400, overlap_tokens=120, min_final_tokens=1)
        assert cfg.overlap_tokens == 120

    def test_min_final_equals_1(self):
        cfg = ChunkerConfig(max_tokens=400, overlap_tokens=80, min_final_tokens=1)
        assert cfg.min_final_tokens == 1

    def test_min_final_equals_max_tokens(self):
        cfg = ChunkerConfig(max_tokens=400, overlap_tokens=80, min_final_tokens=400)
        assert cfg.min_final_tokens == 400

    def test_max_tokens_at_1(self):
        # overlap range: [round(1*0.10), round(1*0.30)] = [0, 0] — both round to 0
        # so overlap_tokens must be 0; min_final must be 1..1
        cfg = ChunkerConfig(max_tokens=1, overlap_tokens=0, min_final_tokens=1)
        assert cfg.max_tokens == 1

    def test_max_tokens_at_upper_bound(self):
        # 8192 — overlap 10%=819, 30%=2458; pick 1000
        cfg = ChunkerConfig(max_tokens=8192, overlap_tokens=1000, min_final_tokens=50)
        assert cfg.max_tokens == 8192


# ---------------------------------------------------------------------------
# ChunkerConfig — invalid max_tokens
# ---------------------------------------------------------------------------

class TestChunkerConfigInvalidMaxTokens:
    def test_zero_max_tokens_raises(self):
        with pytest.raises(ChunkerConfigError, match="max_tokens"):
            ChunkerConfig(max_tokens=0, overlap_tokens=0, min_final_tokens=1)

    def test_negative_max_tokens_raises(self):
        with pytest.raises(ChunkerConfigError, match="max_tokens"):
            ChunkerConfig(max_tokens=-1, overlap_tokens=0, min_final_tokens=1)

    def test_above_8192_max_tokens_raises(self):
        with pytest.raises(ChunkerConfigError, match="max_tokens"):
            ChunkerConfig(max_tokens=8193, overlap_tokens=820, min_final_tokens=1)


# ---------------------------------------------------------------------------
# ChunkerConfig — invalid overlap_tokens
# ---------------------------------------------------------------------------

class TestChunkerConfigInvalidOverlap:
    def test_overlap_below_10_percent_raises(self):
        # 10% of 800 = 80; supply 79 → below floor
        with pytest.raises(ChunkerConfigError, match="overlap_tokens"):
            ChunkerConfig(max_tokens=800, overlap_tokens=79, min_final_tokens=100)

    def test_overlap_above_30_percent_raises(self):
        # 30% of 800 = 240; supply 241 → above ceiling
        with pytest.raises(ChunkerConfigError, match="overlap_tokens"):
            ChunkerConfig(max_tokens=800, overlap_tokens=241, min_final_tokens=100)

    def test_overlap_zero_when_range_excludes_zero_raises(self):
        # For max_tokens=800, valid range is [80, 240]; 0 is invalid
        with pytest.raises(ChunkerConfigError, match="overlap_tokens"):
            ChunkerConfig(max_tokens=800, overlap_tokens=0, min_final_tokens=100)

    def test_negative_overlap_raises(self):
        with pytest.raises(ChunkerConfigError, match="overlap_tokens"):
            ChunkerConfig(max_tokens=800, overlap_tokens=-1, min_final_tokens=100)


# ---------------------------------------------------------------------------
# ChunkerConfig — invalid min_final_tokens
# ---------------------------------------------------------------------------

class TestChunkerConfigInvalidMinFinal:
    def test_zero_min_final_raises(self):
        with pytest.raises(ChunkerConfigError, match="min_final_tokens"):
            ChunkerConfig(max_tokens=800, overlap_tokens=150, min_final_tokens=0)

    def test_negative_min_final_raises(self):
        with pytest.raises(ChunkerConfigError, match="min_final_tokens"):
            ChunkerConfig(max_tokens=800, overlap_tokens=150, min_final_tokens=-5)

    def test_min_final_above_max_tokens_raises(self):
        with pytest.raises(ChunkerConfigError, match="min_final_tokens"):
            ChunkerConfig(max_tokens=800, overlap_tokens=150, min_final_tokens=801)


# ---------------------------------------------------------------------------
# Chunker construction
# ---------------------------------------------------------------------------

class TestChunkerConstruction:
    def test_valid_config_constructs(self):
        cfg = ChunkerConfig()
        c = Chunker(cfg)
        assert c.config is cfg

    def test_uses_default_tokenizer_when_none(self):
        c = Chunker(ChunkerConfig())
        assert c.tokenizer is _DEFAULT_TOKENIZER

    def test_custom_tokenizer_is_stored(self):
        class FakeTok:
            def encode(self, text: str) -> list[int]:
                return list(range(len(text)))
            def decode(self, tokens: list[int]) -> str:
                return " " * len(tokens)

        tok = FakeTok()
        c = Chunker(ChunkerConfig(), tokenizer=tok)
        assert c.tokenizer is tok

    def test_invalid_config_raises_on_construction(self):
        with pytest.raises(ChunkerConfigError):
            Chunker(ChunkerConfig(max_tokens=800, overlap_tokens=79, min_final_tokens=100))


# ---------------------------------------------------------------------------
# Tokenizer (cl100k_base)
# ---------------------------------------------------------------------------

class TestDefaultTokenizer:
    def test_encode_returns_list_of_ints(self):
        tokens = _DEFAULT_TOKENIZER.encode("hello world")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) > 0

    def test_empty_string_returns_empty(self):
        assert _DEFAULT_TOKENIZER.encode("") == []

    def test_decode_roundtrip(self):
        text = "The quick brown fox."
        tokens = _DEFAULT_TOKENIZER.encode(text)
        decoded = _DEFAULT_TOKENIZER.decode(tokens)
        assert decoded == text

    def test_count_tokens_helper(self):
        c = Chunker(ChunkerConfig())
        n = c.count_tokens("hello world")
        # cl100k_base: "hello" + " world" = 2 tokens
        assert n == 2

    def test_count_tokens_empty(self):
        c = Chunker(ChunkerConfig())
        assert c.count_tokens("") == 0
