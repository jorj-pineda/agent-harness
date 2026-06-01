"""SSE streaming endpoint tests (slice 9c).

`GET /chat/stream` wraps the same `run_turn` as `POST /chat` and emits
`tool_start` / `tool_end` events as the loop dispatches tools, then a
`turn_done` event carrying the full envelope. Pre-flight failures
(404/403/400) surface as real HTTP status codes before the stream opens.
"""

from __future__ import annotations

import json
from pathlib import Path

from providers.base import ToolCall

from .conftest import Harness, make_response


def _parse_sse(text: str) -> list[dict[str, str]]:
    """Parse an SSE body into a list of {event, data} dicts."""
    events: list[dict[str, str]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        event: dict[str, str] = {}
        for line in block.splitlines():
            if line.startswith("event:"):
                event["event"] = line[len("event:") :].strip()
            elif line.startswith("data:"):
                event["data"] = line[len("data:") :].strip()
        events.append(event)
    return events


def _make_session(harness: Harness, user_id: str, workspace: Path | None = None) -> str:
    body: dict[str, str] = {"user_id": user_id}
    if workspace is not None:
        body["workspace_root"] = str(workspace)
    resp = harness.client.post("/sessions", json=body)
    assert resp.status_code == 200, resp.text
    return str(resp.json()["session_id"])


def test_stream_final_answer_only(harness: Harness) -> None:
    session_id = _make_session(harness, "u-s1")
    harness.provider.script(make_response(content="hello there"))

    resp = harness.client.get(
        "/chat/stream",
        params={"user_id": "u-s1", "session_id": session_id, "message": "hi"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    assert [e["event"] for e in events] == ["turn_done"]
    done = json.loads(events[0]["data"])["response"]
    assert done["answer"] == "hello there"
    assert done["provider"] == "scripted"


def test_stream_emits_tool_then_done(harness: Harness, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("x = 1\n", encoding="utf-8")
    session_id = _make_session(harness, "u-s2", workspace=repo)

    harness.provider.script(
        make_response(
            tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "calc.py"})],
            finish_reason="tool_use",
        ),
        make_response(content="calc.py sets x to 1"),
    )

    resp = harness.client.get(
        "/chat/stream",
        params={"user_id": "u-s2", "session_id": session_id, "message": "read calc.py"},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    kinds = [e["event"] for e in events]
    assert kinds == ["tool_start", "tool_end", "turn_done"]

    start = json.loads(events[0]["data"])
    assert start["tool"] == "read_file"
    assert start["arguments"] == {"path": "calc.py"}

    end = json.loads(events[1]["data"])
    assert end["tool"] == "read_file"
    assert end["error"] is None
    assert "x = 1" in end["result_snippet"]
    assert end["latency_ms"] >= 0.0

    done = json.loads(events[2]["data"])["response"]
    assert done["answer"] == "calc.py sets x to 1"
    assert len(done["tool_calls"]) == 1


def test_stream_out_of_scope_is_policy_turn(harness: Harness) -> None:
    session_id = _make_session(harness, "u-s3")
    resp = harness.client.get(
        "/chat/stream",
        params={
            "user_id": "u-s3",
            "session_id": session_id,
            "message": "delete all tests in the repo",
        },
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert [e["event"] for e in events] == ["turn_done"]
    done = json.loads(events[0]["data"])["response"]
    assert done["provider"] == "policy"
    assert done["escalated"] is True


def test_stream_404_for_unknown_session(harness: Harness) -> None:
    resp = harness.client.get(
        "/chat/stream",
        params={"user_id": "u", "session_id": "nope", "message": "hi"},
    )
    assert resp.status_code == 404


def test_stream_403_for_user_mismatch(harness: Harness) -> None:
    session_id = _make_session(harness, "owner")
    resp = harness.client.get(
        "/chat/stream",
        params={"user_id": "intruder", "session_id": session_id, "message": "hi"},
    )
    assert resp.status_code == 403


def test_stream_400_for_unknown_provider(harness: Harness) -> None:
    session_id = _make_session(harness, "u-s4")
    resp = harness.client.get(
        "/chat/stream",
        params={
            "user_id": "u-s4",
            "session_id": session_id,
            "message": "hi",
            "provider": "does-not-exist",
        },
    )
    assert resp.status_code == 400
