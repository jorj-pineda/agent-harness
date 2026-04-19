"""Provider package: public types, protocols, and backend factory.

Higher layers import from here only. They never import a concrete backend
module, so adding a new backend is a single-file change: implement the
protocol, add a case to the factory.
"""

from __future__ import annotations

from typing import Any

from .anthropic import AnthropicProvider
from .base import (
    ChatMessage,
    ChatProvider,
    Embedder,
    FinishReason,
    ProviderError,
    ProviderResponse,
    Role,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from .ollama import OllamaProvider
from .openai import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "ChatMessage",
    "ChatProvider",
    "Embedder",
    "FinishReason",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderError",
    "ProviderResponse",
    "Role",
    "TokenUsage",
    "ToolCall",
    "ToolSpec",
    "create_chat_provider",
    "create_embedder",
]


def create_chat_provider(name: str, **kwargs: Any) -> ChatProvider:
    """Return a ChatProvider for the named backend.

    Unknown names raise ValueError — callers should validate config up front.
    """
    match name:
        case "ollama":
            return OllamaProvider(**kwargs)
        case "anthropic":
            return AnthropicProvider(**kwargs)
        case "openai":
            return OpenAIProvider(**kwargs)
        case _:
            raise ValueError(f"Unknown chat provider: {name!r}")


def create_embedder(name: str, **kwargs: Any) -> Embedder:
    """Return an Embedder for the named backend.

    Only Ollama currently exposes embeddings; Anthropic has no first-party
    embedding API and OpenAI's is routed separately when we add it.
    """
    match name:
        case "ollama":
            return OllamaProvider(**kwargs)
        case _:
            raise ValueError(f"Unknown embedder: {name!r}")
