"""Opt-in, bounded inputs for a single trusted diagnostic; never application settings.

Model-visible messages/tool arguments may be sensitive. This stores plaintext, not
redacted content or backend output. Bounds are stored-byte/count limits, not RSS,
CPU, hostile-code isolation or a crash-atomic recording transaction.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from harness.trace import request_identity
from providers.base import (
    ChatMessage,
    ChatProvider,
    ConversationProvider,
    ProviderResponse,
    ToolSpec,
)

ChatRequestPath = Literal[
    "/api/chat",
    "/v1/chat/completions",
    "/inference/go/v1/chat/completions",
    "/inference/go/openai/v1/chat/completions",
]
_CHAT_PATHS = (
    "/api/chat",
    "/v1/chat/completions",
    "/inference/go/v1/chat/completions",
    "/inference/go/openai/v1/chat/completions",
)


class CaptureLimits(BaseModel):
    max_file_bytes: int = Field(default=1024 * 1024, gt=0, le=1024 * 1024, strict=True)
    max_total_bytes: int = Field(default=8 * 1024 * 1024, gt=0, le=8 * 1024 * 1024, strict=True)
    max_requests: int = Field(default=12, gt=0, le=12, strict=True)


class RequestCapture:
    """One provider/transport pair, sequential calls, a fresh owner-only directory."""

    def __init__(
        self,
        directory: Path,
        *,
        limits: CaptureLimits | None = None,
        chat_path: ChatRequestPath = "/api/chat",
        lazy: bool = False,
    ) -> None:
        if chat_path not in _CHAT_PATHS:
            raise ValueError("Unsupported diagnostic chat request path")
        if not directory.is_absolute() or directory.resolve() != directory:
            raise ValueError("Capture directory must be a canonical absolute path")
        self.directory = directory
        self.limits = limits or CaptureLimits()
        self._chat_path = chat_path
        self.requests = 0
        self.stored_bytes = 0
        self._active: int | None = None
        self._wire_written = False
        self._failed = False
        self._allocated = False
        if not lazy:
            self._allocate()

    def _allocate(self) -> None:
        if self._allocated:
            return
        # Recheck delayed parent/path resolution, then refuse existing data.
        # Lazy startup/GET review may refer to an existing directory, but an
        # actual chat can never reopen, resume or overwrite that recording.
        if self.directory.resolve() != self.directory:
            raise ValueError("Capture directory must be a canonical absolute path")
        self.directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        self._allocated = True

    def _write(self, suffix: str, data: bytes) -> None:
        if self._active is None or self._failed:
            raise ValueError("No active healthy capture")
        try:
            if (
                len(data) > self.limits.max_file_bytes
                or self.stored_bytes + len(data) > self.limits.max_total_bytes
            ):
                raise ValueError("Diagnostic input capture byte bound exceeded")
            path = self.directory / f"request-{self._active:04d}-{suffix}.json"
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            self.stored_bytes += len(data)
        except BaseException:
            # A partial file is retained, and this recorder cannot dispatch more work.
            self._failed = True
            raise

    def begin(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None,
        temperature: float,
        max_tokens: int | None,
    ) -> None:
        if self._failed or self._active is not None:
            raise ValueError("Failed or overlapping diagnostic capture")
        if self.requests >= self.limits.max_requests:
            self._failed = True
            raise ValueError("Diagnostic request count bound exceeded")
        identity = request_identity(messages, tools or [], max_tokens)
        if identity is None:
            self._failed = True
            raise ValueError("Unsupported normalized diagnostic input")
        self.requests += 1
        self._active = self.requests
        self._wire_written = False
        try:
            self._allocate()
            payload = {
                "kind": "normalized-diagnostic-input-v1",
                "sequence": self.requests,
                "identity": identity.model_dump(),
                "messages": [message.model_dump() for message in messages],
                # Preserve None versus an explicitly empty tools argument too.
                "tools": None if tools is None else [tool.model_dump() for tool in tools],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._write("normalized", data)
        except BaseException:
            self._failed = True
            self._active = None
            raise

    def wire(self, request: httpx.Request) -> None:
        try:
            if self._active is None or self._failed or self._wire_written:
                raise ValueError("Missing or duplicate normalized/wire capture pairing")
            if request.method != "POST" or request.url.path != self._chat_path:
                raise ValueError("Diagnostic capture accepts chat requests only")
            # Retain exact original bytes, never headers, URL, extensions or output.
            if not isinstance(json.loads(request.content), dict):
                raise ValueError("Diagnostic wire input must be a JSON object")
            self._write("wire", request.content)
            self._wire_written = True
        except BaseException:
            self._failed = True
            raise

    def end(self, *, returned: bool) -> None:
        missing = returned and (self._failed or not self._wire_written)
        self._active = None
        if missing:
            self._failed = True
            raise ValueError("Provider returned without paired wire input evidence")


class CapturedProvider:
    def __init__(self, inner: ChatProvider, capture: RequestCapture) -> None:
        self.inner = inner
        self.name = inner.name
        self.capture = capture
        self._conversation_set = False
        self._conversation_id: str | None = None

    def for_conversation(self, conversation_id: str) -> ChatProvider:
        inner = (
            self.inner.for_conversation(conversation_id)
            if isinstance(self.inner, ConversationProvider)
            else self.inner
        )
        return _CapturedConversationView(self, inner, conversation_id)

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        return await self._chat(
            self.inner,
            None,
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def _chat(
        self,
        inner: ChatProvider,
        conversation_id: str | None,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None,
        temperature: float,
        max_tokens: int | None,
    ) -> ProviderResponse:
        # Binding is a pure view. The first actual call claims this phase's
        # recorder, preventing later views from mixing conversations or scopes.
        if self._conversation_set and self._conversation_id != conversation_id:
            raise ValueError("A diagnostic capture cannot mix conversations")
        self._conversation_set = True
        self._conversation_id = conversation_id
        self.capture.begin(messages, tools, temperature, max_tokens)
        returned = False
        try:
            response = await inner.chat(
                messages, tools=tools, temperature=temperature, max_tokens=max_tokens
            )
            returned = True
            return response
        finally:
            self.capture.end(returned=returned)

    async def aclose(self) -> None:
        close: Any = getattr(self.inner, "aclose", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result


class _CapturedConversationView:
    """Preserve provider-local routing; the original wrapper owns resource cleanup."""

    def __init__(self, parent: CapturedProvider, inner: ChatProvider, conversation_id: str) -> None:
        self.name = parent.name
        self._parent = parent
        self._inner = inner
        self._conversation_id = conversation_id

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        return await self._parent._chat(
            self._inner,
            self._conversation_id,
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class CapturedTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport, capture: RequestCapture) -> None:
        self.inner = inner
        self.capture = capture

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.capture.wire(request)
        # Forward the very same object and timeout extensions. Do not read,
        # buffer, normalize or retry the response in this observer.
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()
