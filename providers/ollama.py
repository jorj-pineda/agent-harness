"""Ollama provider: wraps /api/chat and /api/embed via httpx.

Ollama normalizes across local models (Gemma 4, Llama 3.x, Qwen, etc.) into a
single wire format, so per-model tool-calling quirks are Ollama's concern.
From here we consume its documented response shape.

One divergence from OpenAI-style APIs: Ollama does not return a per-call id on
tool_calls. We synthesize one so the harness can thread tool results back.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import httpx

from .base import (
    ChatMessage,
    FinishReason,
    ProviderError,
    ProviderResponse,
    TokenUsage,
    ToolCall,
    ToolSpec,
)


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        host: str,
        model: str,
        embed_model: str,
        *,
        timeout_seconds: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._embed_model = embed_model
        self._client = httpx.AsyncClient(
            base_url=host.rstrip("/"),
            timeout=timeout_seconds,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [_message_to_ollama(m) for m in messages],
            "stream": False,
            "options": options,
        }
        if tools:
            body["tools"] = [_tool_to_ollama(t) for t in tools]

        start = time.perf_counter()
        try:
            response = await self._client.post("/api/chat", json=body)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama chat request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return _response_from_ollama(
            response.json(), fallback_model=self._model, latency_ms=latency_ms
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.post(
                "/api/embed",
                json={"model": self._embed_model, "input": texts},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama embed request failed: {exc}") from exc
        data: dict[str, Any] = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise ProviderError(f"Ollama embed returned unexpected payload: {data!r}")
        return [list(map(float, vec)) for vec in embeddings]


def _message_to_ollama(m: ChatMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {"function": {"name": tc.name, "arguments": tc.arguments}} for tc in m.tool_calls
        ]
    return out


def _tool_to_ollama(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        },
    }


def _response_from_ollama(
    payload: dict[str, Any], *, fallback_model: str, latency_ms: float
) -> ProviderResponse:
    message = payload.get("message", {}) or {}
    raw_tool_calls = message.get("tool_calls") or []
    tool_calls = [
        ToolCall(
            id=f"call_{uuid.uuid4().hex[:12]}",
            name=tc["function"]["name"],
            arguments=tc["function"].get("arguments", {}) or {},
        )
        for tc in raw_tool_calls
    ]

    finish: FinishReason
    if tool_calls:
        finish = "tool_use"
    else:
        done_reason = payload.get("done_reason", "stop")
        finish = "length" if done_reason == "length" else "stop"

    return ProviderResponse(
        content=message.get("content", "") or "",
        tool_calls=tool_calls,
        finish_reason=finish,
        usage=TokenUsage(
            prompt_tokens=payload.get("prompt_eval_count"),
            completion_tokens=payload.get("eval_count"),
        ),
        model=payload.get("model") or fallback_model,
        latency_ms=latency_ms,
        raw=payload,
    )
