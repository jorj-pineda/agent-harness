"""Anthropic provider: wraps the Messages API via the official SDK.

Uses AsyncAnthropic with an injectable httpx client so tests intercept HTTP
traffic with MockTransport and never hit the live API.

Two divergences from the Ollama/OpenAI-style shape we normalize at this
boundary: `system` is a top-level parameter (not a message role), and
responses are structured content blocks rather than a flat string.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from anthropic import APIError, AsyncAnthropic
from anthropic.types import Message, TextBlock, ToolUseBlock

from .base import (
    ChatMessage,
    FinishReason,
    ProviderError,
    ProviderResponse,
    TokenUsage,
    ToolCall,
    ToolSpec,
)

_STOP_REASON_MAP: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_use",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        default_max_tokens: int = 4096,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._default_max_tokens = default_max_tokens
        self._client = AsyncAnthropic(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
            http_client=http_client,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        system, converted = _split_system(messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted,
            "max_tokens": max_tokens or self._default_max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [_tool_to_anthropic(t) for t in tools]

        start = time.perf_counter()
        try:
            message = await self._client.messages.create(**kwargs)
        except APIError as exc:
            raise ProviderError(f"Anthropic chat request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return _response_from_anthropic(message, fallback_model=self._model, latency_ms=latency_ms)


def _split_system(messages: list[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            if m.content:
                system_parts.append(m.content)
            continue
        converted.append(_message_to_anthropic(m))
    return "\n\n".join(system_parts), converted


def _message_to_anthropic(m: ChatMessage) -> dict[str, Any]:
    # Tool results are encoded as user-role messages with a tool_result block.
    if m.role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id or "",
                    "content": m.content,
                }
            ],
        }

    if m.tool_calls:
        blocks: list[dict[str, Any]] = []
        if m.content:
            blocks.append({"type": "text", "text": m.content})
        for tc in m.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
            )
        return {"role": m.role, "content": blocks}

    return {"role": m.role, "content": m.content}


def _tool_to_anthropic(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters_schema,
    }


def _response_from_anthropic(
    message: Message, *, fallback_model: str, latency_ms: float
) -> ProviderResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            arguments = dict(block.input) if isinstance(block.input, dict) else {}
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=arguments))

    stop_reason = message.stop_reason or "end_turn"
    finish: FinishReason = _STOP_REASON_MAP.get(stop_reason, "stop")

    return ProviderResponse(
        content="".join(text_parts),
        tool_calls=tool_calls,
        finish_reason=finish,
        usage=TokenUsage(
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
        ),
        model=message.model or fallback_model,
        latency_ms=latency_ms,
        raw=message.model_dump(),
    )
