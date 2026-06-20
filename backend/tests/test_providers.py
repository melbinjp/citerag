"""Unit tests for backend/providers — LLMProvider protocol and GeminiProvider."""
from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure a dummy API key is present before importing anything that might read it
os.environ.setdefault("GOOGLE_API_KEY", "test-api-key")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from providers import LLMProvider, GeminiProvider, get_provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def collect(ait: AsyncIterator[str]) -> list[str]:
    """Drain an async iterator into a list."""
    return [chunk async for chunk in ait]


# ---------------------------------------------------------------------------
# LLMProvider protocol
# ---------------------------------------------------------------------------

class TestLLMProviderProtocol:
    """Verify that the Protocol is structural and runtime-checkable."""

    def test_gemini_satisfies_protocol(self):
        provider = GeminiProvider(api_key="key")
        assert isinstance(provider, LLMProvider)

    def test_custom_class_satisfies_protocol(self):
        class MyProvider:
            name = "custom"

            async def generate(
                self, prompt: str, *, timeout_s: float, stream: bool
            ) -> AsyncIterator[str]:
                yield "ok"

        assert isinstance(MyProvider(), LLMProvider)

    def test_class_missing_generate_does_not_satisfy_protocol(self):
        class BadProvider:
            name = "bad"
            # no generate method

        assert not isinstance(BadProvider(), LLMProvider)

    def test_class_missing_name_does_not_satisfy_protocol(self):
        class BadProvider:
            async def generate(self, prompt, *, timeout_s, stream):
                yield ""

        assert not isinstance(BadProvider(), LLMProvider)


# ---------------------------------------------------------------------------
# get_provider factory
# ---------------------------------------------------------------------------

class TestGetProvider:
    def test_default_returns_gemini(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": ""}, clear=False):
            # remove LLM_PROVIDER so the fallback default kicks in
            env = os.environ.copy()
            env.pop("LLM_PROVIDER", None)
            with patch.dict(os.environ, env, clear=True):
                provider = get_provider()
        assert provider.name == "gemini"
        assert isinstance(provider, GeminiProvider)

    def test_explicit_name_gemini(self):
        provider = get_provider("gemini")
        assert isinstance(provider, GeminiProvider)

    def test_env_var_selects_provider(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}):
            provider = get_provider()
        assert isinstance(provider, GeminiProvider)

    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider("nonexistent_provider")

    def test_case_insensitive(self):
        provider = get_provider("GEMINI")
        assert isinstance(provider, GeminiProvider)


# ---------------------------------------------------------------------------
# GeminiProvider construction
# ---------------------------------------------------------------------------

class TestGeminiProviderConstruction:
    def test_init_with_explicit_key(self):
        provider = GeminiProvider(api_key="explicit-key")
        assert provider.name == "gemini"

    def test_init_uses_env_key(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key"}):
            provider = GeminiProvider()
        assert provider.name == "gemini"

    def test_init_no_key_raises(self):
        env = {k: v for k, v in os.environ.items() if k != "GOOGLE_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
                GeminiProvider()

    def test_custom_model_name(self):
        provider = GeminiProvider(api_key="key", model_name="gemini-pro")
        assert provider._model == "gemini-pro"

    def test_default_model_from_env(self):
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-custom"}):
            provider = GeminiProvider(api_key="key")
        assert provider._model == "gemini-custom"


# ---------------------------------------------------------------------------
# GeminiProvider.generate — non-streaming
# ---------------------------------------------------------------------------

class TestGeminiProviderNonStreaming:
    def _make_provider(self) -> GeminiProvider:
        return GeminiProvider(api_key="key")

    def _mock_response(self, text: str) -> MagicMock:
        resp = MagicMock()
        resp.text = text
        return resp

    @pytest.mark.asyncio
    async def test_non_streaming_yields_full_text(self):
        provider = self._make_provider()
        mock_resp = self._mock_response("Hello world")

        with patch.object(
            provider._client.aio.models,
            "generate_content",
            new=AsyncMock(return_value=mock_resp),
        ):
            gen = await provider.generate("test prompt", timeout_s=10, stream=False)
            chunks = await collect(gen)

        assert chunks == ["Hello world"]

    @pytest.mark.asyncio
    async def test_non_streaming_strips_whitespace(self):
        provider = self._make_provider()
        mock_resp = self._mock_response("  trimmed  ")

        with patch.object(
            provider._client.aio.models,
            "generate_content",
            new=AsyncMock(return_value=mock_resp),
        ):
            gen = await provider.generate("prompt", timeout_s=10, stream=False)
            chunks = await collect(gen)

        assert chunks == ["trimmed"]

    @pytest.mark.asyncio
    async def test_non_streaming_timeout_raises(self):
        provider = self._make_provider()

        async def slow_generate(*a, **kw):
            await asyncio.sleep(999)

        with patch.object(
            provider._client.aio.models,
            "generate_content",
            new=AsyncMock(side_effect=slow_generate),
        ):
            gen = await provider.generate("prompt", timeout_s=0.01, stream=False)
            with pytest.raises(asyncio.TimeoutError):
                await collect(gen)

    @pytest.mark.asyncio
    async def test_non_streaming_api_error_raises_runtime_error(self):
        from google.genai import errors as genai_errors

        provider = self._make_provider()

        err = genai_errors.APIError(500, "internal error", None)

        with patch.object(
            provider._client.aio.models,
            "generate_content",
            new=AsyncMock(side_effect=err),
        ):
            gen = await provider.generate("prompt", timeout_s=10, stream=False)
            with pytest.raises(RuntimeError, match="API error"):
                await collect(gen)


# ---------------------------------------------------------------------------
# GeminiProvider.generate — streaming
# ---------------------------------------------------------------------------

class TestGeminiProviderStreaming:
    def _make_provider(self) -> GeminiProvider:
        return GeminiProvider(api_key="key")

    def _async_iter(self, items):
        """Build an async iterable from a list of items."""

        async def _gen():
            for item in items:
                yield item

        return _gen()

    @pytest.mark.asyncio
    async def test_streaming_yields_chunks(self):
        provider = self._make_provider()

        chunk_a = MagicMock()
        chunk_a.text = "Hello "
        chunk_b = MagicMock()
        chunk_b.text = "world"

        with patch.object(
            provider._client.aio.models,
            "generate_content_stream",
            new=AsyncMock(return_value=self._async_iter([chunk_a, chunk_b])),
        ):
            gen = await provider.generate("prompt", timeout_s=10, stream=True)
            chunks = await collect(gen)

        assert chunks == ["Hello ", "world"]

    @pytest.mark.asyncio
    async def test_streaming_skips_empty_chunks(self):
        provider = self._make_provider()

        chunk_a = MagicMock()
        chunk_a.text = "data"
        chunk_b = MagicMock()
        chunk_b.text = ""  # empty — should be skipped
        chunk_c = MagicMock()
        chunk_c.text = "more"

        with patch.object(
            provider._client.aio.models,
            "generate_content_stream",
            new=AsyncMock(
                return_value=self._async_iter([chunk_a, chunk_b, chunk_c])
            ),
        ):
            gen = await provider.generate("prompt", timeout_s=10, stream=True)
            chunks = await collect(gen)

        assert chunks == ["data", "more"]

    @pytest.mark.asyncio
    async def test_streaming_timeout_raises(self):
        provider = self._make_provider()

        async def slow_stream(*a, **kw):
            await asyncio.sleep(999)

        with patch.object(
            provider._client.aio.models,
            "generate_content_stream",
            new=AsyncMock(side_effect=slow_stream),
        ):
            gen = await provider.generate("prompt", timeout_s=0.01, stream=True)
            with pytest.raises(asyncio.TimeoutError):
                await collect(gen)


# ---------------------------------------------------------------------------
# GeminiProvider fallback behaviour
# ---------------------------------------------------------------------------

class TestGeminiProviderFallback:
    """When the primary model returns 429/503, the provider tries fallbacks."""

    def _make_provider(self) -> GeminiProvider:
        return GeminiProvider(api_key="key")

    @pytest.mark.asyncio
    async def test_falls_back_on_429(self):
        from google.genai import errors as genai_errors

        provider = self._make_provider()
        call_count = 0
        ok_resp = MagicMock()
        ok_resp.text = "fallback answer"

        async def flaky(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise genai_errors.APIError(429, "rate limited", None)
            return ok_resp

        with patch.object(
            provider._client.aio.models,
            "generate_content",
            new=AsyncMock(side_effect=flaky),
        ):
            gen = await provider.generate("prompt", timeout_s=10, stream=False)
            chunks = await collect(gen)

        assert "fallback answer" in chunks
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_all_models_fail_raises_runtime_error(self):
        from google.genai import errors as genai_errors

        provider = self._make_provider()

        async def always_rate_limited(*a, **kw):
            raise genai_errors.APIError(429, "rate limited", None)

        with patch.object(
            provider._client.aio.models,
            "generate_content",
            new=AsyncMock(side_effect=always_rate_limited),
        ):
            gen = await provider.generate("prompt", timeout_s=10, stream=False)
            with pytest.raises(RuntimeError, match="capacity errors"):
                await collect(gen)
