"""Explicit, per-turn cancellation independent of transport task cancellation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class TurnCancelled(Exception):
    """Owned work has settled after an explicit cancellation request."""


class TurnCancellation:
    """Event-loop-owned signal; repeated requests do not interrupt cleanup."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.closed = False

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def request(self) -> bool:
        if self.closed:
            return False
        self._event.set()
        return True

    def close(self) -> None:
        self.closed = True

    async def run(self, operation: Callable[[], Awaitable[T]]) -> T:
        if self.requested:
            raise TurnCancelled
        work = asyncio.ensure_future(operation())
        signal = asyncio.create_task(self._event.wait())
        try:
            await asyncio.wait((work, signal), return_when=asyncio.FIRST_COMPLETED)
            if work.done():
                return work.result()
            await _cancel_and_settle(work)
            raise TurnCancelled
        except asyncio.CancelledError:
            await _cancel_and_settle(work)
            raise
        finally:
            signal.cancel()
            await asyncio.gather(signal, return_exceptions=True)


async def _cancel_and_settle(work: asyncio.Future[T]) -> None:
    work.cancel()
    while not work.done():
        try:
            await asyncio.shield(work)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    if not work.cancelled():
        work.exception()
