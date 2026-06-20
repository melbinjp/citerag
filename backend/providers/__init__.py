"""LLM provider abstraction for the RAG PDF Chatbot.

Public surface
--------------
- ``LLMProvider`` — structural ``Protocol`` that every provider must satisfy.
- ``GeminiProvider`` — default hosted provider (Google Gemini via google-genai).
- ``get_provider(name)`` — factory that returns the configured provider instance.

Usage
-----
    from backend.providers import get_provider

    provider = get_provider("gemini")           # uses GOOGLE_API_KEY env var
    async for token in await provider.generate(prompt, timeout_s=30, stream=True):
        print(token, end="", flush=True)
"""
from __future__ import annotations

import os
from typing import AsyncIterator, Protocol, runtime_checkable

from .gemini import GeminiProvider


@runtime_checkable
class LLMProvider(Protocol):
    """Structural protocol for all LLM provider implementations.

    Any class that exposes a ``name`` attribute and an async ``generate``
    method with the matching signature satisfies this protocol — no explicit
    inheritance required.
    """

    #: Short identifier for this provider, e.g. ``"gemini"``.
    name: str

    async def generate(
        self,
        prompt: str,
        *,
        timeout_s: float,
        stream: bool,
    ) -> AsyncIterator[str]:
        """Generate a response for *prompt*.

        Parameters
        ----------
        prompt:
            Fully assembled prompt text.
        timeout_s:
            Hard deadline in seconds.  Implementations must raise
            ``asyncio.TimeoutError`` if the deadline is breached.
        stream:
            If *True*, yield tokens incrementally; if *False*, yield the
            complete response as a single string.

        Yields
        ------
        str
            One or more text fragments that together form the full response.
        """
        ...  # pragma: no cover


# Registry of provider name -> factory callable.
# Add new providers here as they become available.
_REGISTRY: dict[str, type[LLMProvider]] = {
    "gemini": GeminiProvider,  # type: ignore[type-abstract]
}


def get_provider(name: str | None = None) -> LLMProvider:
    """Return a fully initialised LLM provider.

    The *name* is resolved with the following priority:

    1. The ``name`` argument, if provided.
    2. The ``LLM_PROVIDER`` environment variable.
    3. ``"gemini"`` as the hard default.

    Parameters
    ----------
    name:
        Provider identifier (case-insensitive).  Must be one of the keys in
        the provider registry (currently ``"gemini"``).

    Returns
    -------
    LLMProvider
        A ready-to-use provider instance.

    Raises
    ------
    ValueError
        If the resolved provider name is not in the registry.
    """
    resolved = (name or os.getenv("LLM_PROVIDER", "gemini")).lower().strip()
    provider_cls = _REGISTRY.get(resolved)
    if provider_cls is None:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown LLM provider {resolved!r}. "
            f"Available providers: {available}."
        )
    return provider_cls()


__all__ = [
    "LLMProvider",
    "GeminiProvider",
    "get_provider",
]
