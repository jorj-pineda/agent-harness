from __future__ import annotations

import pytest

from providers import (
    AnthropicProvider,
    ChatProvider,
    Embedder,
    OllamaProvider,
    OpenAIProvider,
    create_chat_provider,
    create_embedder,
)


def test_create_chat_provider_dispatches_by_name() -> None:
    ollama = create_chat_provider(
        "ollama", host="http://fake", model="gemma4", embed_model="nomic-embed-text"
    )
    anthropic = create_chat_provider("anthropic", api_key="test", model="claude-sonnet-4-6")
    openai = create_chat_provider("openai", api_key="test", model="gpt-4o")

    assert isinstance(ollama, OllamaProvider)
    assert isinstance(anthropic, AnthropicProvider)
    assert isinstance(openai, OpenAIProvider)

    assert isinstance(ollama, ChatProvider)
    assert isinstance(anthropic, ChatProvider)
    assert isinstance(openai, ChatProvider)


def test_create_embedder_dispatches_to_ollama() -> None:
    embedder = create_embedder(
        "ollama", host="http://fake", model="gemma4", embed_model="nomic-embed-text"
    )
    assert isinstance(embedder, OllamaProvider)
    assert isinstance(embedder, Embedder)


def test_unknown_chat_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown chat provider"):
        create_chat_provider("cohere")


def test_unknown_embedder_raises() -> None:
    with pytest.raises(ValueError, match="Unknown embedder"):
        create_embedder("anthropic")
