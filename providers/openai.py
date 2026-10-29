"""OpenAI provider: wraps chat completions via the official SDK.

Uses AsyncOpenAI with an injectable httpx client so tests intercept HTTP
traffic with MockTransport and never hit the live API.

One divergence from the other providers: OpenAI serializes tool-call
arguments as a JSON string (not a parsed object). We json.loads it at the
boundary so downstream code sees a dict.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

import httpx
from openai import APIError, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageFunctionToolCall
from pydantic import ValidationError

from .base import (
    ChatMessage,
    ChatProvider,
    ContinuationState,
    FinishReason,
    ProviderError,
    ProviderResponse,
    TokenUsage,
    ToolCall,
    ToolSpec,
    reject_unsupported_continuation,
)


def compatible_transport_options(
    transport: str, *, replay_reasoning_content: bool = False
) -> dict[str, Any]:
    """Only public configured options; never keys or per-conversation values."""
    if transport not in {"generic", "opencode_go"}:
        raise ValueError("Unknown compatible transport")
    options: dict[str, Any] = {}
    if transport == "opencode_go":
        options = {
            "default_headers": {"User-Agent": "agent-harness/0.1"},
            "conversation_header": "x-opencode-session",
            "max_retries": 0,
        }
    if replay_reasoning_content:
        options["replay_reasoning_content"] = True
    return options


_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "content_filter",
}


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        provider_name: str = "openai",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        default_headers: Mapping[str, str] | None = None,
        conversation_header: str | None = None,
        replay_reasoning_content: bool = False,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = provider_name
        self._model = model
        self._conversation_header = conversation_header
        self._replay_reasoning_content = replay_reasoning_content
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
        )

    async def aclose(self) -> None:
        await self._client.close()

    def for_conversation(self, conversation_id: str) -> ChatProvider:
        if self._conversation_header is None:
            return self
        if (
            not conversation_id
            or len(conversation_id) > 256
            or any(ord(char) < 33 or ord(char) > 126 for char in conversation_id)
        ):
            raise ValueError("Conversation ID must be bounded printable ASCII without spaces")
        return _ConversationView(self, {self._conversation_header: conversation_id})

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        if self._conversation_header is not None:
            raise ProviderError("Bind a conversation before calling this configured transport")
        return await self._chat(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )

    async def _chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None,
        temperature: float,
        max_tokens: int | None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> ProviderResponse:
        if not self._replay_reasoning_content:
            reject_unsupported_continuation(messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                _message_to_openai(m, replay_reasoning_content=self._replay_reasoning_content)
                for m in messages
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = [_tool_to_openai(t) for t in tools]
        if extra_headers is not None:
            kwargs["extra_headers"] = extra_headers

        start = time.perf_counter()
        try:
            completion = await self._client.chat.completions.create(**kwargs)
        except APIError as exc:
            raise ProviderError(f"OpenAI chat request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return _response_from_openai(
            completion,
            fallback_model=self._model,
            latency_ms=latency_ms,
            replay_reasoning_content=self._replay_reasoning_content,
        )


class _ConversationView:
    """Request-local headers over one owned SDK client, safe for concurrent sessions."""

    def __init__(self, parent: OpenAIProvider, headers: Mapping[str, str]) -> None:
        self.name = parent.name
        self._parent = parent
        self._headers = headers

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        return await self._parent._chat(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=self._headers,
        )


def _message_to_openai(m: ChatMessage, *, replay_reasoning_content: bool = False) -> dict[str, Any]:
    if m.continuation is not None and (not replay_reasoning_content or m.role != "assistant"):
        raise ProviderError("This message cannot replay assistant continuation state")
    if m.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": m.tool_call_id or "",
            "content": m.content,
        }

    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in m.tool_calls
        ]
    if m.continuation is not None:
        out["reasoning_content"] = m.continuation.text
    return out


def _tool_to_openai(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        },
    }


def _response_from_openai(
    completion: ChatCompletion,
    *,
    fallback_model: str,
    latency_ms: float,
    replay_reasoning_content: bool = False,
) -> ProviderResponse:
    choice = completion.choices[0]
    message = choice.message
    continuation = None
    if replay_reasoning_content:
        value = (message.model_extra or {}).get("reasoning_content")
        if value is not None:
            try:
                continuation = ContinuationState(format="openai-reasoning-content-v1", text=value)
            except (ValidationError, ValueError):
                raise ProviderError("Invalid or oversized assistant continuation state") from None

    tool_calls: list[ToolCall] = []
    for tc in message.tool_calls or []:
        if not isinstance(tc, ChatCompletionMessageFunctionToolCall):
            continue
        raw_args = tc.function.arguments or "{}"
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError:
            arguments = {"_raw": raw_args}
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))

    finish: FinishReason = _FINISH_REASON_MAP.get(choice.finish_reason or "", "error")

    usage = completion.usage
    return ProviderResponse(
        content=message.content or "",
        tool_calls=tool_calls,
        finish_reason=finish,
        usage=TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        ),
        model=completion.model or fallback_model,
        latency_ms=latency_ms,
        raw=completion.model_dump(),
        continuation=continuation,
    )
