"""Evaluation-only first-input enforcement, not a launch or approval mechanism."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import httpx

from evals.heldout_environment import local_endpoint
from evals.heldout_guard import _read_regular
from evals.heldout_inputs import normalized_input
from evals.heldout_staging import verify_capture
from providers.base import ChatMessage, ProviderResponse, ToolSpec
from providers.ollama import OllamaProvider


@dataclass(frozen=True)
class InputPins:
    normalized: bytes
    wire: bytes

    def __post_init__(self) -> None:
        if (
            not self.normalized
            or not self.wire
            or max(len(self.normalized), len(self.wire)) > 2_000_000
        ):
            raise ValueError("Invalid first-input byte bounds")


def load_pins(
    capture: Path, *, original_directory: Path, expected_index_sha256: str, ordinal: int
) -> InputPins:
    index = verify_capture(
        capture,
        expected_index_sha256=expected_index_sha256,
        original_directory=original_directory.resolve(),
    )
    if not 1 <= ordinal <= len(index.cells):
        raise ValueError("Unknown fixed cohort ordinal")
    cell = index.cells[ordinal - 1]
    contents = []
    for name in (f"{ordinal:02d}-normalized.json", f"{ordinal:02d}-wire.json"):
        raw, mode = _read_regular(capture / name, 2_000_000)
        if mode != 0o600 or hashlib.sha256(raw).hexdigest() != cell.files[name]:
            raise ValueError("First-input pins changed after bundle verification")
        contents.append(raw)
    return InputPins(*contents)


class PinnedTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport, expected: bytes) -> None:
        self.inner, self.expected = inner, expected
        self.attempted_requests = 0
        self.failed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.failed:
            raise ValueError("First-input transport is latched closed")
        try:
            if request.method != "POST" or request.url.path != "/api/chat":
                raise ValueError("Unexpected inference route")
            if self.attempted_requests == 0 and request.content != self.expected:
                raise ValueError("Frozen first wire input differs; no request forwarded")
            self.attempted_requests += 1
            response = await self.inner.handle_async_request(request)
            if response.status_code >= 300:
                self.failed = True
            return response
        except BaseException:
            self.failed = True
            raise

    async def aclose(self) -> None:
        self.failed = True
        await self.inner.aclose()


class PinnedOllamaProvider:
    name = "ollama"

    def __init__(
        self,
        *,
        pins: InputPins,
        host: str,
        model: str,
        num_ctx: int,
        think: bool,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport,
    ) -> None:
        endpoint = local_endpoint(host)
        self.pins = pins
        self.transport = PinnedTransport(transport, pins.wire)
        self.adapter = OllamaProvider(
            host=endpoint,
            model=model,
            embed_model="unused",
            num_ctx=num_ctx,
            think=think,
            timeout_seconds=timeout_seconds,
            transport=self.transport,
        )
        self.started = False
        self.failed = False
        self.closed = False
        self.inflight = False

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        if self.failed or self.closed or self.inflight:
            self.failed = True
            raise ValueError("First-input provider is closed or has an unresolved request")
        self.inflight = True
        try:
            if (
                not self.started
                and normalized_input(messages, tools, temperature, max_tokens)
                != self.pins.normalized
            ):
                raise ValueError("Frozen normalized first input differs; adapter not called")
            self.started = True
            response = await self.adapter.chat(
                messages, tools=tools, temperature=temperature, max_tokens=max_tokens
            )
            raw = response.raw
            if (
                not isinstance(raw, dict)
                or not isinstance(raw.get("model"), str)
                or not raw["model"].strip()
                or raw.get("done") is not True
            ):
                raise ValueError(
                    "Native Ollama response lacks explicit model or terminal done=true"
                )
            return response
        except BaseException:
            self.failed = True
            raise
        finally:
            self.inflight = False

    async def aclose(self) -> None:
        if not self.closed:
            self.closed = True
            await self.adapter.aclose()
