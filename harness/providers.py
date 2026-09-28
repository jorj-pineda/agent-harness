"""Configured chat backends shared by the API and real coding evaluator."""

from __future__ import annotations

from providers import create_chat_provider
from providers.base import ChatProvider

from .config import Settings


def configured_provider_names(settings: Settings) -> tuple[str, ...]:
    names = ["ollama"]
    if settings.anthropic_api_key:
        names.append("anthropic")
    if settings.openai_api_key:
        names.append("openai")
    if settings.openai_compatible_base_url:
        names.append("openai_compatible")
    return tuple(names)


def build_configured_provider(name: str, settings: Settings) -> ChatProvider:
    available = configured_provider_names(settings)
    if name not in available:
        raise ValueError(f"Provider {name!r} is not configured (available: {available})")
    timeout = float(settings.request_timeout_seconds)
    if name == "ollama":
        return create_chat_provider(
            "ollama",
            host=settings.ollama_host,
            model=settings.ollama_model,
            embed_model=settings.ollama_embed_model,
            timeout_seconds=timeout,
        )
    if name == "anthropic":
        assert settings.anthropic_api_key is not None
        return create_chat_provider(
            "anthropic",
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            timeout_seconds=timeout,
        )
    if name == "openai":
        assert settings.openai_api_key is not None
        return create_chat_provider(
            "openai",
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout_seconds=timeout,
        )
    assert settings.openai_compatible_base_url is not None
    assert settings.openai_compatible_model is not None
    return create_chat_provider(
        "openai",
        api_key=settings.openai_compatible_api_key or "not-required",
        model=settings.openai_compatible_model,
        base_url=settings.openai_compatible_base_url,
        provider_name="openai_compatible",
        timeout_seconds=timeout,
    )


def provider_endpoint(name: str, settings: Settings) -> str:
    if name == "ollama":
        return settings.ollama_host
    if name == "openai_compatible":
        return settings.openai_compatible_base_url or "unconfigured"
    return f"{name}:default"


def configured_model(name: str, settings: Settings) -> str:
    if name == "ollama":
        return settings.ollama_model
    if name == "anthropic":
        return settings.anthropic_model
    if name == "openai":
        return settings.openai_model
    if name == "openai_compatible" and settings.openai_compatible_model:
        return settings.openai_compatible_model
    raise ValueError(f"Unknown or unconfigured provider: {name}")
