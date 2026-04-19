from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tests._cassette import CassetteMissError, CassetteTransport


def _stub_transport(status: int, payload: dict[str, object]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload, request=request)

    return httpx.MockTransport(handler)


async def test_record_then_replay_roundtrip(tmp_path: Path) -> None:
    cassette_path = tmp_path / "roundtrip.json"

    recorder = CassetteTransport(
        cassette_path,
        mode="record",
        inner=_stub_transport(200, {"hello": "world"}),
    )
    async with httpx.AsyncClient(transport=recorder, base_url="http://fake") as client:
        first = await client.post("/chat", json={"q": "ping"})
    assert first.status_code == 200
    assert first.json() == {"hello": "world"}
    recorder.save()

    assert cassette_path.exists()
    saved = json.loads(cassette_path.read_text(encoding="utf-8"))
    assert len(saved) == 1
    assert saved[0]["response"]["body"] == {"json": {"hello": "world"}}

    replayer = CassetteTransport(cassette_path, mode="replay")
    async with httpx.AsyncClient(transport=replayer, base_url="http://fake") as client:
        second = await client.post("/chat", json={"q": "ping"})
    assert second.status_code == 200
    assert second.json() == {"hello": "world"}


async def test_replay_miss_raises(tmp_path: Path) -> None:
    cassette_path = tmp_path / "empty.json"
    cassette_path.write_text("[]", encoding="utf-8")

    transport = CassetteTransport(cassette_path, mode="replay")
    async with httpx.AsyncClient(transport=transport, base_url="http://fake") as client:
        with pytest.raises(CassetteMissError):
            await client.post("/chat", json={"q": "unseen"})


async def test_body_differences_produce_distinct_keys(tmp_path: Path) -> None:
    cassette_path = tmp_path / "distinct.json"

    recorder = CassetteTransport(
        cassette_path,
        mode="record",
        inner=_stub_transport(200, {"ok": True}),
    )
    async with httpx.AsyncClient(transport=recorder, base_url="http://fake") as client:
        await client.post("/chat", json={"q": "a"})
        await client.post("/chat", json={"q": "b"})
    recorder.save()

    saved = json.loads(cassette_path.read_text(encoding="utf-8"))
    assert len({row["key"] for row in saved}) == 2
