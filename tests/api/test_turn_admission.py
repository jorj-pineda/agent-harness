from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from providers.base import ToolCall

from .conftest import Harness, make_response
from .test_stream import _parse_sse


@pytest.mark.parametrize("first_transport", ["http", "stream"])
def test_http_and_stream_share_turn_admission(
    harness: Harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_transport: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    sessions = [
        harness.client.post(
            "/sessions",
            json={
                "user_id": "dev",
                "workspace_root": str(repo),
            },
        ).json()["session_id"]
        for _ in range(2)
    ]
    started, release = threading.Event(), threading.Event()
    original_chat = harness.provider.chat

    async def holding_chat(*args, **kwargs):
        started.set()
        assert await asyncio.to_thread(release.wait, 5)
        return await original_chat(*args, **kwargs)

    monkeypatch.setattr(harness.provider, "chat", holding_chat)
    harness.provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="first",
                    name="write_file",
                    arguments={"path": "first.py", "content": "first = 1\n"},
                )
            ]
        ),
        make_response(content="First complete"),
        make_response(
            tool_calls=[
                ToolCall(
                    id="second",
                    name="write_file",
                    arguments={"path": "second.py", "content": "second = 2\n"},
                )
            ]
        ),
        make_response(content="Second complete"),
    )

    def request(transport, session):
        payload = {"user_id": "dev", "session_id": session, "message": "Update the module"}
        if transport == "http":
            response = harness.client.post("/chat", json=payload)
            assert response.status_code == 200, response.text
            return response.json()
        response = harness.client.get("/chat/stream", params=payload)
        assert response.status_code == 200, response.text
        events = _parse_sse(response.text)
        assert events[-1]["event"] == "turn_done"
        return json.loads(events[-1]["data"])["response"]

    second_transport = "stream" if first_transport == "http" else "http"
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(request, first_transport, sessions[0])
        try:
            assert started.wait(2)
            blocked = request(second_transport, sessions[1])
            assert blocked["completion_status"] == "blocked"
            assert "workspace overlaps" in blocked["completion_reason"]
            assert blocked["tool_calls"] == []
            assert not (repo / "second.py").exists()
            assert harness.provider.calls == []
            # A busy response must not save the other turn's mutable transcript.
            for sid in sessions:
                saved = harness.client.get(
                    f"/sessions/{sid}", params={"user_id": "dev"}
                ).json()
                assert saved["session"]["turns"] == []
                assert saved["responses"] == []
        finally:
            release.set()
            completed = first.result(timeout=5)
    assert completed["completion_status"] == "completed"
    assert completed["workspace_changes"]["added"] == ["first.py"]
    retry = request(second_transport, sessions[1])
    assert retry["completion_status"] == "completed"
    assert retry["workspace_changes"]["added"] == ["second.py"]
    assert [d["path"] for d in retry["workspace_changes"]["diffs"]] == ["second.py"]
