from __future__ import annotations

import pytest

from harness.router import ProviderNotFoundError, ProviderRouter
from providers.base import ChatMessage, ProviderResponse, ToolSpec


class StubProvider:
    """Minimal `ChatProvider` — just a name; `chat` is never called here."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        raise AssertionError("router tests should not invoke chat()")


def test_resolve_with_no_name_returns_default() -> None:
    ollama = StubProvider("ollama")
    anthropic = StubProvider("anthropic")
    router = ProviderRouter({"ollama": ollama, "anthropic": anthropic}, default="ollama")

    assert router.resolve() is ollama


def test_resolve_by_name_returns_the_matching_provider() -> None:
    ollama = StubProvider("ollama")
    anthropic = StubProvider("anthropic")
    router = ProviderRouter({"ollama": ollama, "anthropic": anthropic}, default="ollama")

    assert router.resolve("anthropic") is anthropic


def test_resolve_unknown_name_raises_provider_not_found() -> None:
    router = ProviderRouter({"ollama": StubProvider("ollama")}, default="ollama")

    with pytest.raises(ProviderNotFoundError) as excinfo:
        router.resolve("openai")
    assert "openai" in str(excinfo.value)


def test_empty_mapping_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="at least one provider"):
        ProviderRouter({}, default="ollama")


def test_default_must_be_present_in_mapping() -> None:
    with pytest.raises(ValueError, match="default provider"):
        ProviderRouter({"ollama": StubProvider("ollama")}, default="anthropic")


def test_names_returns_sorted_registered_keys() -> None:
    router = ProviderRouter(
        {
            "ollama": StubProvider("ollama"),
            "anthropic": StubProvider("anthropic"),
            "openai": StubProvider("openai"),
        },
        default="ollama",
    )

    assert router.names() == ["anthropic", "ollama", "openai"]


def test_default_property_exposes_configured_default() -> None:
    router = ProviderRouter(
        {"ollama": StubProvider("ollama"), "anthropic": StubProvider("anthropic")},
        default="anthropic",
    )

    assert router.default == "anthropic"


def test_router_is_insulated_from_later_mutations_to_input_mapping() -> None:
    ollama = StubProvider("ollama")
    mapping = {"ollama": ollama}
    router = ProviderRouter(mapping, default="ollama")

    mapping["anthropic"] = StubProvider("anthropic")

    assert router.names() == ["ollama"]
    with pytest.raises(ProviderNotFoundError):
        router.resolve("anthropic")
