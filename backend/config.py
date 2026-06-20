"""
backend/config.py

Application configuration loader with range validation.

All settings are read from environment variables (with sensible defaults).
Call :func:`load_config` to obtain a validated, immutable :class:`AppConfig`
instance.  A :class:`ConfigError` is raised if any value falls outside its
permitted range.

Configuration variables
-----------------------
LLM_PROVIDER          : str   default "gemini"
EMBEDDING_MODEL       : str   default "BAAI/bge-m3"
CHUNK_MAX_TOKENS      : int   default 800    range [1, 8192]
CHUNK_OVERLAP_TOKENS  : int   default 150    range [round(max*0.1), round(max*0.3)]
CHUNK_MIN_FINAL_TOKENS: int   default 100    range [1, max]
DEFAULT_TOP_K         : int   default 5      range [1, 100]
RERANK_ENABLED        : bool  default False
RERANK_TIMEOUT_S      : float default 2.0    range > 0
GENERATION_TIMEOUT_S  : float default 30.0   range [1, 120]
QDRANT_URL            : str   default "http://qdrant:6333"
QDRANT_COLLECTION     : str   default "corpus"
CORPUS_DIR            : str   default "./data/corpus"
INGESTION_ERROR_LOG   : str   default "./data/ingestion_errors.jsonl"

Requirements: 4.1, 8.6, 14.3, 14.4
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when a configuration value falls outside its permitted range."""


# ---------------------------------------------------------------------------
# AppConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    """Immutable, validated application configuration.

    Instances should be created only via :func:`load_config`, which performs
    all range validation.
    """

    llm_provider: str               # LLM_PROVIDER
    embedding_model: str            # EMBEDDING_MODEL
    chunk_max_tokens: int           # CHUNK_MAX_TOKENS,   range [1, 8192]
    chunk_overlap_tokens: int       # CHUNK_OVERLAP_TOKENS, range [10%..30% of max]
    chunk_min_final_tokens: int     # CHUNK_MIN_FINAL_TOKENS, range [1..max]
    default_top_k: int              # DEFAULT_TOP_K,      range [1, 100]
    rerank_enabled: bool            # RERANK_ENABLED
    rerank_timeout_s: float         # RERANK_TIMEOUT_S,   range > 0
    generation_timeout_s: float     # GENERATION_TIMEOUT_S, range [1, 120]
    qdrant_url: str                 # QDRANT_URL
    qdrant_collection: str          # QDRANT_COLLECTION
    corpus_dir: str                 # CORPUS_DIR
    ingestion_error_log: str        # INGESTION_ERROR_LOG


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config() -> AppConfig:
    """Load and validate all configuration from environment variables.

    Returns
    -------
    AppConfig
        A validated, immutable configuration object.

    Raises
    ------
    ConfigError
        When any numeric value falls outside its permitted range, with a
        descriptive message identifying the failing variable and the range.
    """
    # --- String settings (no range validation) ---
    llm_provider: str = os.environ.get("LLM_PROVIDER", "gemini")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
    qdrant_url: str = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    qdrant_collection: str = os.environ.get("QDRANT_COLLECTION", "corpus")
    corpus_dir: str = os.environ.get("CORPUS_DIR", "./data/corpus")
    ingestion_error_log: str = os.environ.get(
        "INGESTION_ERROR_LOG", "./data/ingestion_errors.jsonl"
    )

    # --- CHUNK_MAX_TOKENS: range [1, 8192] ---
    chunk_max_tokens: int = _parse_int("CHUNK_MAX_TOKENS", default=800)
    if not (1 <= chunk_max_tokens <= 8192):
        raise ConfigError(
            f"CHUNK_MAX_TOKENS must be in [1, 8192], got {chunk_max_tokens}."
        )

    # --- CHUNK_OVERLAP_TOKENS: range [round(max*0.1), round(max*0.3)] ---
    overlap_floor = round(chunk_max_tokens * 0.1)
    overlap_ceiling = round(chunk_max_tokens * 0.3)
    chunk_overlap_tokens: int = _parse_int("CHUNK_OVERLAP_TOKENS", default=150)
    if not (overlap_floor <= chunk_overlap_tokens <= overlap_ceiling):
        raise ConfigError(
            f"CHUNK_OVERLAP_TOKENS must be in "
            f"[{overlap_floor}, {overlap_ceiling}] "
            f"(10%–30% of CHUNK_MAX_TOKENS={chunk_max_tokens}), "
            f"got {chunk_overlap_tokens}."
        )

    # --- CHUNK_MIN_FINAL_TOKENS: range [1, chunk_max_tokens] ---
    chunk_min_final_tokens: int = _parse_int("CHUNK_MIN_FINAL_TOKENS", default=100)
    if not (1 <= chunk_min_final_tokens <= chunk_max_tokens):
        raise ConfigError(
            f"CHUNK_MIN_FINAL_TOKENS must be in [1, {chunk_max_tokens}] "
            f"(1..CHUNK_MAX_TOKENS), got {chunk_min_final_tokens}."
        )

    # --- DEFAULT_TOP_K: range [1, 100] ---
    default_top_k: int = _parse_int("DEFAULT_TOP_K", default=5)
    if not (1 <= default_top_k <= 100):
        raise ConfigError(
            f"DEFAULT_TOP_K must be in [1, 100], got {default_top_k}."
        )

    # --- RERANK_ENABLED: boolean flag ---
    rerank_enabled: bool = os.environ.get("RERANK_ENABLED", "false").strip().lower() in (
        "1", "true", "yes",
    )

    # --- RERANK_TIMEOUT_S: range > 0 ---
    rerank_timeout_s: float = _parse_float("RERANK_TIMEOUT_S", default=2.0)
    if rerank_timeout_s <= 0:
        raise ConfigError(
            f"RERANK_TIMEOUT_S must be greater than 0, got {rerank_timeout_s}."
        )

    # --- GENERATION_TIMEOUT_S: range [1, 120] ---
    generation_timeout_s: float = _parse_float("GENERATION_TIMEOUT_S", default=30.0)
    if not (1.0 <= generation_timeout_s <= 120.0):
        raise ConfigError(
            f"GENERATION_TIMEOUT_S must be in [1, 120], got {generation_timeout_s}."
        )

    return AppConfig(
        llm_provider=llm_provider,
        embedding_model=embedding_model,
        chunk_max_tokens=chunk_max_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        chunk_min_final_tokens=chunk_min_final_tokens,
        default_top_k=default_top_k,
        rerank_enabled=rerank_enabled,
        rerank_timeout_s=rerank_timeout_s,
        generation_timeout_s=generation_timeout_s,
        qdrant_url=qdrant_url,
        qdrant_collection=qdrant_collection,
        corpus_dir=corpus_dir,
        ingestion_error_log=ingestion_error_log,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_int(env_var: str, *, default: int) -> int:
    """Parse an environment variable as an integer.

    Parameters
    ----------
    env_var:
        Name of the environment variable to read.
    default:
        Value to use when the variable is unset or empty.

    Returns
    -------
    int

    Raises
    ------
    ConfigError
        When the variable is set but cannot be parsed as a valid integer.
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(
            f"{env_var} must be a valid integer, got {raw!r}."
        )


def _parse_float(env_var: str, *, default: float) -> float:
    """Parse an environment variable as a float.

    Parameters
    ----------
    env_var:
        Name of the environment variable to read.
    default:
        Value to use when the variable is unset or empty.

    Returns
    -------
    float

    Raises
    ------
    ConfigError
        When the variable is set but cannot be parsed as a valid float.
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(
            f"{env_var} must be a valid number, got {raw!r}."
        )
