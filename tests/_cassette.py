"""VCR-style cassette transport for httpx.

Intercepts `httpx.AsyncClient` requests and either replays a saved exchange
(CI default) or records a new one from the real backend (developer-initiated).

Why a custom transport instead of a library like vcrpy: vcrpy's httpx support
is fragile across versions, and subclassing `AsyncBaseTransport` is ~100 LOC.
Keeping it in-repo also means cassettes stay readable (parsed JSON bodies),
which matters for a portfolio project — reviewers will open them.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import httpx

CassetteMode = Literal["replay", "record"]


def current_mode() -> CassetteMode:
    raw = os.getenv("CASSETTE_MODE", "replay").lower()
    if raw not in ("replay", "record"):
        raise ValueError(f"CASSETTE_MODE must be 'replay' or 'record', got {raw!r}")
    return raw  # type: ignore[return-value]


def _encode_body(content: bytes) -> dict[str, Any]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return {"b64": base64.b64encode(content).decode("ascii")}
    try:
        return {"json": json.loads(text)}
    except (json.JSONDecodeError, ValueError):
        return {"text": text}


def _decode_body(encoded: dict[str, Any]) -> bytes:
    if "b64" in encoded:
        return base64.b64decode(encoded["b64"])
    if "json" in encoded:
        return json.dumps(encoded["json"]).encode("utf-8")
    return str(encoded.get("text", "")).encode("utf-8")


def _match_key(method: str, url: str, body: bytes) -> str:
    body_hash = hashlib.sha256(body).hexdigest()[:16]
    return f"{method.upper()} {url} {body_hash}"


class CassetteMissError(RuntimeError):
    """Raised when replay mode has no saved interaction matching a request."""


class CassetteTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        path: Path,
        mode: CassetteMode,
        *,
        inner: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.path = path
        self.mode = mode
        self._inner = inner if inner is not None else httpx.AsyncHTTPTransport()
        self._interactions: list[dict[str, Any]] = self._load() if path.exists() else []

    def _load(self) -> list[dict[str, Any]]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._interactions, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = request.content
        key = _match_key(request.method, str(request.url), body)

        if self.mode == "replay":
            for interaction in self._interactions:
                if interaction["key"] == key:
                    return _build_response(interaction["response"])
            raise CassetteMissError(
                f"No recorded interaction for {key!r} in {self.path}. "
                "Re-run with CASSETTE_MODE=record to refresh."
            )

        response = await self._inner.handle_async_request(request)
        response_bytes = await response.aread()
        self._interactions.append(
            {
                "key": key,
                "request": {
                    "method": request.method,
                    "url": str(request.url),
                    "body": _encode_body(body),
                },
                "response": {
                    "status": response.status_code,
                    "headers": {
                        k: v for k, v in response.headers.items() if k.lower() == "content-type"
                    },
                    "body": _encode_body(response_bytes),
                },
            }
        )
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=response_bytes,
            request=request,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


def _build_response(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code=payload["status"],
        headers=payload.get("headers", {}),
        content=_decode_body(payload["body"]),
    )
