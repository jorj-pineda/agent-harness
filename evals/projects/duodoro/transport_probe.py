"""Evaluation-only Ollama transport experiment; not selected by application settings.

Keep the production adapter's request serialization and final normalization. Change
only the wire stream flag, buffer complete terminal output before returning it,
and retain bounded numeric timing evidence outside model-visible content.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, StrictBool, StrictStr

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_FRAMES = 32768
MAX_TOOL_CALLS = 64


class TransportObservation(BaseModel):
    mode: Literal["buffered", "streamed"]
    duration_ms: float = Field(ge=0, allow_inf_nan=False)
    first_body_chunk_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    largest_body_gap_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    body_chunks: int = Field(ge=0)
    response_bytes: int = Field(ge=0)
    frames: int = Field(ge=0)
    terminal_received: bool | None
    outcome: Literal["completed", "http_error", "transport_error", "protocol_error", "cancelled"]


class StreamFunction(BaseModel):
    name: StrictStr = Field(min_length=1)
    arguments: dict[str, Any] | None = None


class StreamTool(BaseModel):
    function: StreamFunction


class StreamMessage(BaseModel):
    content: StrictStr | None = ""
    tool_calls: list[StreamTool] | None = None


class StreamFrame(BaseModel):
    done: StrictBool
    message: StreamMessage | None = None


class StreamAssembly:
    def __init__(self) -> None:
        self.frames = 0
        self.terminal: dict[str, Any] | None = None
        self.content: list[str] = []
        self.tools: list[dict[str, Any]] = []

    def add(self, line: bytes) -> None:
        if not line.strip():
            return
        if self.terminal is not None:
            raise ValueError("Data after terminal frame")
        self.frames += 1
        if self.frames > MAX_FRAMES:
            raise ValueError("Frame limit")
        raw = json.loads(line.decode("utf-8"))
        if not isinstance(raw, dict) or "error" in raw:
            raise ValueError("Invalid stream frame")
        frame = StreamFrame.model_validate(raw)
        if frame.message is not None:
            self.content.append(frame.message.content or "")
            for tool in frame.message.tool_calls or []:
                self.tools.append(
                    {
                        "function": {
                            "name": tool.function.name,
                            "arguments": tool.function.arguments or {},
                        }
                    }
                )
                if len(self.tools) > MAX_TOOL_CALLS:
                    raise ValueError("Tool-call assembly limit")
        if frame.done:
            self.terminal = raw

    def finish(self) -> bytes:
        if self.terminal is None:
            raise ValueError("Missing terminal frame")
        payload = dict(self.terminal)
        payload["message"] = {"content": "".join(self.content), "tool_calls": self.tools}
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class ProbeTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        inner: httpx.AsyncBaseTransport,
        *,
        streamed: bool,
        observe: Callable[[TransportObservation], None],
    ) -> None:
        self.inner = inner
        self.streamed = streamed
        self.observe = observe

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        start = time.perf_counter()
        response: httpx.Response | None = None
        first: float | None = None
        last = start
        largest: float | None = None
        chunks = 0
        size = 0
        assembly = StreamAssembly()
        outcome: Literal[
            "completed", "http_error", "transport_error", "protocol_error", "cancelled"
        ] = "protocol_error"
        try:
            if request.method != "POST" or request.url.path != "/api/chat":
                raise ValueError("Probe accepts chat requests only")
            body = json.loads(request.content)
            if not isinstance(body, dict) or body.get("stream") is not False:
                raise ValueError("Expected production non-streaming body")
            body["stream"] = self.streamed
            headers = {k: v for k, v in request.headers.items() if k != "content-length"}
            wire = httpx.Request(
                request.method,
                request.url,
                headers=headers,
                content=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                extensions=dict(request.extensions),
            )
            response = await self.inner.handle_async_request(wire)
            if response.status_code >= 400:
                outcome = "http_error"
                return httpx.Response(response.status_code, request=request)
            buffered = bytearray()
            pending = bytearray()
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                now = time.perf_counter()
                first = first if first is not None else (now - start) * 1000
                gap = (now - last) * 1000
                largest = gap if largest is None else max(largest, gap)
                last = now
                chunks += 1
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise ValueError("Response byte limit")
                if self.streamed:
                    scan_from = len(pending)
                    pending.extend(chunk)
                    while True:
                        index = pending.find(b"\n", scan_from)
                        if index < 0:
                            break
                        line = bytes(pending[:index])
                        del pending[: index + 1]
                        assembly.add(line)
                        scan_from = 0
                else:
                    buffered.extend(chunk)
            if self.streamed:
                if pending:
                    assembly.add(bytes(pending))
                content = assembly.finish()
            else:
                content = bytes(buffered)
            outcome = "completed"
            return httpx.Response(
                response.status_code,
                headers={"content-type": "application/json"},
                content=content,
                request=request,
            )
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except httpx.HTTPError:
            outcome = "transport_error"
            raise
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            raise httpx.RemoteProtocolError(
                "Probe rejected incomplete or invalid output", request=request
            ) from exc
        finally:
            try:
                if response is not None:
                    await response.aclose()
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            except httpx.HTTPError:
                outcome = "transport_error"
                raise
            except Exception:
                outcome = "protocol_error"
                raise
            finally:
                self.observe(
                    TransportObservation(
                        mode="streamed" if self.streamed else "buffered",
                        duration_ms=(time.perf_counter() - start) * 1000,
                        first_body_chunk_ms=first,
                        largest_body_gap_ms=largest,
                        body_chunks=chunks,
                        response_bytes=size,
                        frames=assembly.frames,
                        terminal_received=(assembly.terminal is not None)
                        if self.streamed
                        else None,
                        outcome=outcome,
                    )
                )

    async def aclose(self) -> None:
        await self.inner.aclose()
