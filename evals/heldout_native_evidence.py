"""Bounded private native payload evidence; no network default or retry."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from evals.heldout_inputs import _save
from evals.heldout_workspaces import _owned_location
from evals.owned_inference_transport import ENDPOINT

MAX_BYTES = 2_000_000
MAX_REQUESTS = 32  # Above the fixed six-iteration protocol, including corrections.


class NativeEvidenceTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport, directory: Path) -> None:
        self.directory = _owned_location(directory)
        if directory.is_symlink() or directory.resolve() != self.directory:
            raise ValueError("Native evidence requires a canonical owned directory")
        self.directory.mkdir(mode=0o700, exist_ok=False)
        self.inner = inner
        self.failed = False
        self.closed = False
        self.inflight = False
        self.requests = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.failed or self.closed or self.inflight or self.requests >= MAX_REQUESTS:
            self.failed = True
            raise ValueError("Native evidence transport is closed or exhausted")
        self.inflight = True
        self.requests += 1
        prefix = self.directory / f"request-{self.requests:02d}"
        response = None
        primary = None
        delegated = False
        body = bytearray()
        body_complete = False
        try:
            if request.method != "POST" or request.url != ENDPOINT:
                raise ValueError("Unexpected native evidence route")
            if any(
                name in request.headers
                for name in ("authorization", "proxy-authorization", "cookie")
            ):
                raise ValueError("Credential-bearing native request refused")
            if len(request.content) > MAX_BYTES:
                raise ValueError("Native request exceeds evidence bound")
            _save(prefix.with_suffix(".request.bin"), request.content)
            delegated = True
            response = await self.inner.handle_async_request(request)
            if response.headers.get("content-encoding", "identity") != "identity":
                raise ValueError("Encoded native response is not supported")
            if response.is_stream_consumed:
                if len(response.content) > MAX_BYTES:
                    raise ValueError("Native response exceeds evidence bound")
                body.extend(response.content)
            else:
                async for chunk in response.aiter_raw(chunk_size=64_000):
                    if len(body) + len(chunk) > MAX_BYTES:
                        raise ValueError("Native response exceeds evidence bound")
                    body.extend(chunk)
            if len(body) > MAX_BYTES:
                raise ValueError("Native response exceeds evidence bound")
            _save(prefix.with_suffix(".response.bin"), bytes(body))
            body_complete = True
            _save(
                prefix.with_suffix(".json"),
                json.dumps(
                    {
                        "status_code": response.status_code,
                        "request_bytes": len(request.content),
                        "response_bytes": len(body),
                        "delegated": True,
                    }
                ).encode(),
            )
            if self.failed or self.closed:
                raise ValueError("Native evidence submission invalidated")
            return httpx.Response(
                response.status_code, headers=response.headers, content=bytes(body)
            )
        except BaseException as error:
            primary = error
            self.failed = True
            try:
                if body and not body_complete:
                    _save(prefix.with_suffix(".partial-response.bin"), bytes(body))
                _save(
                    prefix.with_suffix(".failure.json"),
                    json.dumps(
                        {
                            "error_type": type(error).__name__,
                            "delegated": delegated,
                            "response_body_complete": body_complete,
                        }
                    ).encode(),
                )
            except BaseException as recording_error:
                error.add_note(
                    f"Native failure recording also failed: {type(recording_error).__name__}"
                )
            raise
        finally:
            if response is not None:
                try:
                    await response.aclose()
                except BaseException as closing_error:
                    self.failed = True
                    if primary is None:
                        try:
                            _save(
                                prefix.with_suffix(".close-failure.json"),
                                json.dumps({"error_type": type(closing_error).__name__}).encode(),
                            )
                        except BaseException as recording_error:
                            closing_error.add_note(
                                "Close failure recording also failed: "
                                f"{type(recording_error).__name__}"
                            )
                        raise
                    primary.add_note(
                        f"Native response close also failed: {type(closing_error).__name__}"
                    )

            self.inflight = False

    async def aclose(self) -> None:
        if not self.closed:
            self.closed = True
            self.failed = True
            await self.inner.aclose()
