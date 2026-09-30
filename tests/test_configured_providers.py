from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from api.server import _build_providers
from harness.config import Settings
from harness.providers import (
    build_configured_provider,
    configured_model,
    configured_provider_names,
    provider_endpoint,
)
from providers.base import ChatMessage, ToolSpec
from providers.ollama import OllamaProvider
from providers.openai import OpenAIProvider


def test_compatible_configuration_is_shared_by_application_and_evaluator() -> None:
    settings = Settings(
        _env_file=None,
        default_provider="openai_compatible",
        openai_compatible_base_url="http://localhost:1234/v1/",
        openai_compatible_model="small-coder",
    )
    assert configured_provider_names(settings) == ("ollama", "openai_compatible")
    assert provider_endpoint("openai_compatible", settings) == "http://localhost:1234/v1"
    assert configured_model("openai_compatible", settings) == "small-coder"
    api_provider = _build_providers(settings)["openai_compatible"]
    eval_provider = build_configured_provider("openai_compatible", settings)
    assert isinstance(api_provider, OpenAIProvider)
    assert isinstance(eval_provider, OpenAIProvider)
    assert api_provider.name == eval_provider.name == "openai_compatible"


@pytest.mark.parametrize(
    "values",
    [
        {"openai_compatible_base_url": "http://localhost:1234/v1"},
        {"openai_compatible_model": "small-coder"},
        {
            "openai_compatible_base_url": "http://user:secret@localhost:1234/v1",
            "openai_compatible_model": "small-coder",
        },
        {
            "openai_compatible_base_url": "http://localhost:1234/v1?key=secret",
            "openai_compatible_model": "small-coder",
        },
    ],
)
def test_invalid_compatible_configuration_fails_before_provider_creation(
    values: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)


def test_unconfigured_compatible_provider_is_unavailable() -> None:
    empty = Settings(_env_file=None, openai_compatible_base_url="", openai_compatible_model="")
    assert "openai_compatible" not in configured_provider_names(empty)
    with pytest.raises(ValueError, match="not configured"):
        build_configured_provider("openai_compatible", empty)


async def test_compatible_wire_uses_custom_endpoint_model_and_tool_schema() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "small-coder",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(
        api_key="local-token",
        model="small-coder",
        base_url="http://localhost:1234/v1",
        provider_name="openai_compatible",
        http_client=client,
    )
    try:
        result = await provider.chat(
            [ChatMessage(role="user", content="fix it")],
            tools=[
                ToolSpec(name="read_file", description="Read", parameters_schema={"type": "object"})
            ],
        )
    finally:
        await provider.aclose()
    assert result.content == "done"
    assert provider.name == "openai_compatible"
    assert str(captured[0].url) == "http://localhost:1234/v1/chat/completions"
    body = json.loads(captured[0].content)
    assert body["model"] == "small-coder"
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert captured[0].headers["authorization"] == "Bearer local-token"


def test_ollama_generation_settings_reach_provider() -> None:
    provider = build_configured_provider(
        "ollama", Settings(_env_file=None, ollama_num_ctx=8192, ollama_think=False)
    )
    assert isinstance(provider, OllamaProvider)
    assert provider._num_ctx == 8192
    assert provider._think is False
    defaults = Settings(_env_file=None)
    assert defaults.ollama_num_ctx is None
    assert defaults.ollama_think is None
