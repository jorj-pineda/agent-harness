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
from typing import Any

import httpx
from openai import APIError, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageFunctionToolCall

from .base import (
    ChatMessage,
    FinishReason,
    ProviderError,
    ProviderResponse,
    TokenUsage,
    ToolCall,
    ToolSpec,
)

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
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(
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
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [_message_to_openai(m) for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = [_tool_to_openai(t) for t in tools]

        start = time.perf_counter()
        try:
            completion = await self._client.chat.completions.create(**kwargs)
        except APIError as exc:
            raise ProviderError(f"OpenAI chat request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return _response_from_openai(completion, fallback_model=self._model, latency_ms=latency_ms)


def _message_to_openai(m: ChatMessage) -> dict[str, Any]:
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
    completion: ChatCompletion, *, fallback_model: str, latency_ms: float
) -> ProviderResponse:
    choice = completion.choices[0]
    message = choice.message

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

    finish: FinishReason = _FINISH_REASON_MAP.get(choice.finish_reason or "stop", "stop")

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
    )
