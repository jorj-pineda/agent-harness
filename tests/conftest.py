from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from tests._cassette import CassetteTransport, current_mode

CASSETTES_DIR = Path(__file__).parent / "cassettes"


@pytest.fixture
def cassette(request: pytest.FixtureRequest) -> Callable[..., CassetteTransport]:
    """Factory fixture: `transport = cassette("ollama_basic_chat")`.

    Replay mode by default; set `CASSETTE_MODE=record` to refresh from a live
    backend. In record mode, the cassette is written on test teardown.
    """

    transports: list[CassetteTransport] = []

    def _make(name: str, *, inner: httpx.AsyncBaseTransport | None = None) -> CassetteTransport:
        path = CASSETTES_DIR / f"{name}.json"
        transport = CassetteTransport(path, current_mode(), inner=inner)
        transports.append(transport)
        return transport

    yield _make

    if current_mode() == "record":
        for t in transports:
            t.save()
