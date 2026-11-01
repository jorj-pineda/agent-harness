"""Owned-endpoint submission boundary; caller still requires freeze/approval gates."""

from __future__ import annotations

from collections.abc import Callable

import httpx

ENDPOINT = httpx.URL("http://127.0.0.1:11447/api/chat")


class OwnedInferenceTransport(httpx.AsyncBaseTransport):
    """No default network delegate, redirects, retries or ownership discovery."""

    def __init__(self, inner: httpx.AsyncBaseTransport, check_owned: Callable[[], None]) -> None:
        self.inner = inner
        self.check_owned = check_owned
        self.failed = False
        self.closed = False
        self.inflight = False
        self.attempted_requests = 0
        self.cleanup_error: str | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.failed or self.closed or self.inflight:
            self.failed = True
            raise ValueError("Owned inference transport is closed or submission unresolved")
        self.inflight = True
        response = None
        try:
            if request.method != "POST" or request.url != ENDPOINT:
                raise ValueError("Inference requires the exact owned loopback chat endpoint")
            self.check_owned()  # Must raise for unknown, dead or foreign ownership.
            self.attempted_requests += 1
            response = await self.inner.handle_async_request(request)
            if response.status_code >= 300:
                raise ValueError("Owned inference HTTP response failed; no retry permitted")
            self.check_owned()
            if self.failed or self.closed:
                raise ValueError("Owned inference transport failed during submission")
            return response
        except BaseException:
            self.failed = True
            if response is not None:
                try:
                    await response.aclose()
                except BaseException as error:
                    self.cleanup_error = type(error).__name__
            raise
        finally:
            self.inflight = False

    async def aclose(self) -> None:
        if not self.closed:
            self.closed = True
            self.failed = True
            await self.inner.aclose()
