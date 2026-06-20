"""
tests/test_config.py

Unit tests for backend/config.py — AppConfig loading and range validation.

Requirements: 3.10, 6.3, 8.8 (via 4.1, 8.6, 14.3, 14.4)
Task: 1.2 Write unit tests for configuration validation
"""

import pytest

# config.py lives at the backend package root; import directly
from config import AppConfig, ConfigError, load_config


# ---------------------------------------------------------------------------
# Helper: isolate env vars so tests do not bleed into each other
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove all config-related env vars before each test."""
    vars_to_clear = [
        "LLM_PROVIDER",
        "EMBEDDING_MODEL",
        "CHUNK_MAX_TOKENS",
        "CHUNK_OVERLAP_TOKENS",
        "CHUNK_MIN_FINAL_TOKENS",
        "DEFAULT_TOP_K",
        "RERANK_ENABLED",
        "RERANK_TIMEOUT_S",
        "GENERATION_TIMEOUT_S",
        "QDRANT_URL",
        "QDRANT_COLLECTION",
        "CORPUS_DIR",
        "INGESTION_ERROR_LOG",
    ]
    for var in vars_to_clear:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestDefaultValues:
    def test_defaults_load_without_error(self):
        cfg = load_config()
        assert isinstance(cfg, AppConfig)

    def test_default_llm_provider(self):
        assert load_config().llm_provider == "gemini"

    def test_default_embedding_model(self):
        assert load_config().embedding_model == "BAAI/bge-m3"

    def test_default_chunk_max_tokens(self):
        assert load_config().chunk_max_tokens == 800

    def test_default_chunk_overlap_tokens(self):
        assert load_config().chunk_overlap_tokens == 150

    def test_default_chunk_min_final_tokens(self):
        assert load_config().chunk_min_final_tokens == 100

    def test_default_top_k(self):
        assert load_config().default_top_k == 5

    def test_default_rerank_enabled(self):
        assert load_config().rerank_enabled is False

    def test_default_rerank_timeout(self):
        assert load_config().rerank_timeout_s == 2.0

    def test_default_generation_timeout(self):
        assert load_config().generation_timeout_s == 30.0

    def test_default_qdrant_url(self):
        assert load_config().qdrant_url == "http://qdrant:6333"

    def test_default_qdrant_collection(self):
        assert load_config().qdrant_collection == "corpus"

    def test_config_is_frozen(self):
        cfg = load_config()
        with pytest.raises((AttributeError, TypeError)):
            cfg.llm_provider = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CHUNK_MAX_TOKENS
# ---------------------------------------------------------------------------


class TestChunkMaxTokens:
    def test_valid_min(self, monkeypatch):
        monkeypatch.setenv("CHUNK_MAX_TOKENS", "1")
        monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "0")   # round(1*0.1)=0
        monkeypatch.setenv("CHUNK_MIN_FINAL_TOKENS", "1")
        cfg = load_config()
        assert cfg.chunk_max_tokens == 1

    def test_valid_max(self, monkeypatch):
        monkeypatch.setenv("CHUNK_MAX_TOKENS", "8192")
        monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "1000")
        monkeypatch.setenv("CHUNK_MIN_FINAL_TOKENS", "50")
        cfg = load_config()
        assert cfg.chunk_max_tokens == 8192

    def test_zero_raises(self, monkeypatch):
        monkeypatch.setenv("CHUNK_MAX_TOKENS", "0")
        with pytest.raises(ConfigError, match="CHUNK_MAX_TOKENS"):
            load_config()

    def test_negative_raises(self, monkeypatch):
        monkeypatch.setenv("CHUNK_MAX_TOKENS", "-1")
        with pytest.raises(ConfigError, match="CHUNK_MAX_TOKENS"):
            load_config()

    def test_above_8192_raises(self, monkeypatch):
        monkeypatch.setenv("CHUNK_MAX_TOKENS", "8193")
        with pytest.raises(ConfigError, match="CHUNK_MAX_TOKENS"):
            load_config()

    def test_non_integer_raises(self, monkeypatch):
        monkeypatch.setenv("CHUNK_MAX_TOKENS", "abc")
        with pytest.raises(ConfigError):
            load_config()


# ---------------------------------------------------------------------------
# CHUNK_OVERLAP_TOKENS
# ---------------------------------------------------------------------------


class TestChunkOverlapTokens:
    def test_valid_at_10_percent(self, monkeypatch):
        # max=400 → floor=round(40)=40
        monkeypatch.setenv("CHUNK_MAX_TOKENS", "400")
        monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "40")
        monkeypatch.setenv("CHUNK_MIN_FINAL_TOKENS", "1")
        cfg = load_config()
        assert cfg.chunk_overlap_tokens == 40

    def test_valid_at_30_percent(self, monkeypatch):
        # max=400 → ceiling=round(120)=120
        monkeypatch.setenv("CHUNK_MAX_TOKENS", "400")
        monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "120")
        monkeypatch.setenv("CHUNK_MIN_FINAL_TOKENS", "1")
        cfg = load_config()
        assert cfg.chunk_overlap_tokens == 120

    def test_below_10_percent_raises(self, monkeypatch):
        # max=800 → floor=80; 79 is below
        monkeypatch.setenv("CHUNK_MAX_TOKENS", "800")
        monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "79")
        with pytest.raises(ConfigError, match="CHUNK_OVERLAP_TOKENS"):
            load_config()

    def test_above_30_percent_raises(self, monkeypatch):
        # max=800 → ceiling=240; 241 is above
        monkeypatch.setenv("CHUNK_MAX_TOKENS", "800")
        monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "241")
        with pytest.raises(ConfigError, match="CHUNK_OVERLAP_TOKENS"):
            load_config()


# ---------------------------------------------------------------------------
# CHUNK_MIN_FINAL_TOKENS
# ---------------------------------------------------------------------------


class TestChunkMinFinalTokens:
    def test_valid_at_1(self, monkeypatch):
        monkeypatch.setenv("CHUNK_MIN_FINAL_TOKENS", "1")
        cfg = load_config()
        assert cfg.chunk_min_final_tokens == 1

    def test_valid_equals_max(self, monkeypatch):
        monkeypatch.setenv("CHUNK_MAX_TOKENS", "400")
        monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "80")
        monkeypatch.setenv("CHUNK_MIN_FINAL_TOKENS", "400")
        cfg = load_config()
        assert cfg.chunk_min_final_tokens == 400

    def test_zero_raises(self, monkeypatch):
        monkeypatch.setenv("CHUNK_MIN_FINAL_TOKENS", "0")
        with pytest.raises(ConfigError, match="CHUNK_MIN_FINAL_TOKENS"):
            load_config()

    def test_above_max_raises(self, monkeypatch):
        monkeypatch.setenv("CHUNK_MAX_TOKENS", "400")
        monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "80")
        monkeypatch.setenv("CHUNK_MIN_FINAL_TOKENS", "401")
        with pytest.raises(ConfigError, match="CHUNK_MIN_FINAL_TOKENS"):
            load_config()


# ---------------------------------------------------------------------------
# DEFAULT_TOP_K
# ---------------------------------------------------------------------------


class TestDefaultTopK:
    def test_valid_min(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_TOP_K", "1")
        assert load_config().default_top_k == 1

    def test_valid_max(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_TOP_K", "100")
        assert load_config().default_top_k == 100

    def test_zero_raises(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_TOP_K", "0")
        with pytest.raises(ConfigError, match="DEFAULT_TOP_K"):
            load_config()

    def test_above_100_raises(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_TOP_K", "101")
        with pytest.raises(ConfigError, match="DEFAULT_TOP_K"):
            load_config()


# ---------------------------------------------------------------------------
# RERANK_ENABLED
# ---------------------------------------------------------------------------


class TestRerankEnabled:
    def test_true_string(self, monkeypatch):
        monkeypatch.setenv("RERANK_ENABLED", "true")
        assert load_config().rerank_enabled is True

    def test_false_string(self, monkeypatch):
        monkeypatch.setenv("RERANK_ENABLED", "false")
        assert load_config().rerank_enabled is False

    def test_1_string(self, monkeypatch):
        monkeypatch.setenv("RERANK_ENABLED", "1")
        assert load_config().rerank_enabled is True

    def test_yes_string(self, monkeypatch):
        monkeypatch.setenv("RERANK_ENABLED", "yes")
        assert load_config().rerank_enabled is True

    def test_uppercase_true(self, monkeypatch):
        monkeypatch.setenv("RERANK_ENABLED", "TRUE")
        assert load_config().rerank_enabled is True


# ---------------------------------------------------------------------------
# GENERATION_TIMEOUT_S
# ---------------------------------------------------------------------------


class TestGenerationTimeoutS:
    def test_valid_at_1(self, monkeypatch):
        monkeypatch.setenv("GENERATION_TIMEOUT_S", "1")
        assert load_config().generation_timeout_s == 1.0

    def test_valid_at_120(self, monkeypatch):
        monkeypatch.setenv("GENERATION_TIMEOUT_S", "120")
        assert load_config().generation_timeout_s == 120.0

    def test_below_1_raises(self, monkeypatch):
        monkeypatch.setenv("GENERATION_TIMEOUT_S", "0.9")
        with pytest.raises(ConfigError, match="GENERATION_TIMEOUT_S"):
            load_config()

    def test_above_120_raises(self, monkeypatch):
        monkeypatch.setenv("GENERATION_TIMEOUT_S", "121")
        with pytest.raises(ConfigError, match="GENERATION_TIMEOUT_S"):
            load_config()

    def test_non_numeric_raises(self, monkeypatch):
        monkeypatch.setenv("GENERATION_TIMEOUT_S", "fast")
        with pytest.raises(ConfigError):
            load_config()


# ---------------------------------------------------------------------------
# RERANK_TIMEOUT_S
# ---------------------------------------------------------------------------


class TestRerankTimeoutS:
    def test_valid_positive(self, monkeypatch):
        monkeypatch.setenv("RERANK_TIMEOUT_S", "5.0")
        assert load_config().rerank_timeout_s == 5.0

    def test_zero_raises(self, monkeypatch):
        monkeypatch.setenv("RERANK_TIMEOUT_S", "0")
        with pytest.raises(ConfigError, match="RERANK_TIMEOUT_S"):
            load_config()

    def test_negative_raises(self, monkeypatch):
        monkeypatch.setenv("RERANK_TIMEOUT_S", "-1.0")
        with pytest.raises(ConfigError, match="RERANK_TIMEOUT_S"):
            load_config()


# ---------------------------------------------------------------------------
# String settings (no range validation, just pass-through)
# ---------------------------------------------------------------------------


class TestStringSettings:
    def test_custom_llm_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        assert load_config().llm_provider == "ollama"

    def test_custom_qdrant_url(self, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", "http://myhost:6333")
        assert load_config().qdrant_url == "http://myhost:6333"

    def test_custom_corpus_dir(self, monkeypatch):
        monkeypatch.setenv("CORPUS_DIR", "/mnt/data/corpus")
        assert load_config().corpus_dir == "/mnt/data/corpus"

    def test_custom_ingestion_error_log(self, monkeypatch):
        monkeypatch.setenv("INGESTION_ERROR_LOG", "/mnt/data/errors.jsonl")
        assert load_config().ingestion_error_log == "/mnt/data/errors.jsonl"
