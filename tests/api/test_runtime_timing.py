"""Review duration persists with the final envelope across both transports."""

import json

import pytest
from fastapi.testclient import TestClient

from providers.base import ToolCall

from .conftest import ScriptedProvider, make_response
from .test_session_persistence import make_app
from .test_stream import _parse_sse


@pytest.mark.parametrize("transport", ["http", "stream"])
def test_review_duration_is_saved_and_replayed_after_restart(tmp_path, transport):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("user = 1\n")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "a.py", "content": "user = 1\nagent = 2\n"},
                )
            ]
        ),
        make_response(content="Done"),
    )
    with TestClient(make_app(tmp_path, provider)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        payload = {
            "user_id": "dev",
            "session_id": sid,
            "message": "Update a.py",
            "request_id": "one",
        }
        if transport == "http":
            result = client.post("/chat", json=payload).json()
        else:
            events = _parse_sse(client.get("/chat/stream", params=payload).text)
            result = json.loads(events[-1]["data"])["response"]
        assert result["workspace_review_ms"] > 0
        assert result["latency_ms"] >= 0
        archive = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        assert archive["responses"][0]["response"] == result
        record = client.get(f"/runs/{result['run_id']}", params={"user_id": "dev"}).json()
        assert record["response"] == result
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as client:
        replay = client.post("/chat", json=payload).json()
        assert replay == result
        assert client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archive
        assert (repo / "a.py").read_text() == "user = 1\nagent = 2\n"
        assert restarted.calls == []
