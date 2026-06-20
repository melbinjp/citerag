"""GeminiProvider — default hosted LLM provider using google-genai."""
from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

from google import genai
from google.genai import errors


class GeminiProvider:
    """LLMProvider implementation backed by Google Gemini via google-genai.

    Uses the same ``genai.Client`` pattern as ``app.py``.  The model name
    is resolved from the ``GEMINI_MODEL`` env var, falling back to the
    production default.  The client is initialised lazily so that unit tests
    that never call ``generate`` do not require a live API key.
    """

    name: str = "gemini"

    #: Primary model to try first, then fallbacks in order.
    _DEFAULT_MODEL = "gemini-2.5-flash"
    _FALLBACK_MODELS: list[str] = ["gemini-2.0-flash", "gemini-1.5-flash"]

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "GeminiProvider requires GOOGLE_API_KEY to be set "
                "(or passed as api_key)."
            )
        self._client = genai.Client(api_key=resolved_key)
        self._model = model_name or os.getenv("GEMINI_MODEL", self._DEFAULT_MODEL)

    async def generate(
        self,
        prompt: str,
        *,
        timeout_s: float = 30.0,
        stream: bool = False,
    ) -> AsyncIterator[str]:
        """Yield text tokens from Gemini.

        Parameters
        ----------
        prompt:
            The fully assembled prompt string to send to the model.
        timeout_s:
            Maximum seconds to wait for the complete response (or first chunk
            when streaming).  Raises ``asyncio.TimeoutError`` on breach.
        stream:
            When *True*, yields tokens as they arrive from the streaming API.
            When *False*, yields the complete response text as a single chunk.

        Yields
        ------
        str
            One or more text fragments.

        Raises
        ------
        asyncio.TimeoutError
            If the provider does not respond within *timeout_s* seconds.
        RuntimeError
            If the provider returns a non-retryable API error.
        """
        return self._generate_impl(prompt, timeout_s=timeout_s, stream=stream)

    async def _generate_impl(
        self,
        prompt: str,
        *,
        timeout_s: float,
        stream: bool,
    ) -> AsyncIterator[str]:
        """Internal async generator that drives the Gemini API call."""
        models_to_try = [self._model] + self._FALLBACK_MODELS
        last_error: Exception | None = None

        for model in models_to_try:
            try:
                if stream:
                    async for token in self._stream(prompt, model, timeout_s):
                        yield token
                else:
                    text = await self._complete(prompt, model, timeout_s)
                    yield text
                return  # success — stop trying fallbacks

            except errors.APIError as exc:
                if exc.code in (429, 503):
                    # Capacity error — try next fallback
                    last_error = exc
                    continue
                # Non-retryable API error
                raise RuntimeError(
                    f"GeminiProvider API error (model={model}, "
                    f"code={exc.code}): {exc.message}"
                ) from exc
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError(
                    f"GeminiProvider timed out after {timeout_s}s "
                    f"(model={model})."
                )
            except Exception as exc:
                raise RuntimeError(
                    f"GeminiProvider unexpected error (model={model}): {exc}"
                ) from exc

        # All models exhausted due to capacity errors
        raise RuntimeError(
            "GeminiProvider: all models returned capacity errors (429/503). "
            f"Last error: {last_error}"
        )

    async def _complete(self, prompt: str, model: str, timeout_s: float) -> str:
        """Non-streaming completion; returns the full response text."""
        response = await asyncio.wait_for(
            self._client.aio.models.generate_content(
                model=model,
                contents=prompt,
            ),
            timeout=timeout_s,
        )
        return (response.text or "").strip()

    async def _stream(
        self, prompt: str, model: str, timeout_s: float
    ) -> AsyncIterator[str]:
        """Streaming completion; yields each text chunk as it arrives."""
        response = await asyncio.wait_for(
            self._client.aio.models.generate_content_stream(
                model=model,
                contents=prompt,
            ),
            timeout=timeout_s,
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text
