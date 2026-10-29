import json

import pytest
from fastapi.testclient import TestClient

from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tests.api.test_continuation import SETTINGS, body, known_provider, submit_and_finish
from tests.api.test_session_persistence import make_app
from tests.api.test_stream import _parse_sse


def test_inspection_creation_validation_and_standard_default(tmp_path):
    with TestClient(make_app(tmp_path, ScriptedProvider())) as api:
        for value in ("true", 1, None):
            assert (
                api.post("/sessions", json={"user_id": "dev", "inspection_only": value}).status_code
                == 422
            )
        created = api.post("/sessions", json={"user_id": "dev"}).json()
        assert created["inspection_only"] is False
        sid = created["session_id"]
        saved = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        assert saved["session"]["inspection_only"] is False


def test_inspection_policy_persists_and_fork_cannot_gain_edit_permission(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "user.py"
    target.write_text("user = 1\n")
    provider = known_provider()
    provider.script(
        make_response(
            prompt_tokens=10,
            completion_tokens=2,
            tool_calls=[ToolCall(id="read", name="read_file", arguments={"path": "user.py"})],
        ),
        make_response(content="user is 1", prompt_tokens=10, completion_tokens=2),
    )
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as api:
        created = api.post(
            "/sessions",
            json={
                "user_id": "dev",
                "workspace_root": str(root),
                "inspection_only": True,
                "checkpoint_turns": True,
            },
        ).json()
        assert created["inspection_only"] is True
        sid = created["session_id"]
        response = api.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Explain"}
        )
        assert response.json()["completion_status"] == "completed"
        saved = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        assert saved["session"]["inspection_only"] is True
        assert saved["responses"][0]["execution_profile"]["inspection_only"] is True
    later = known_provider()
    later.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="write",
                    name="write_file",
                    arguments={"path": "user.py", "content": "overwritten"},
                )
            ]
        )
    )
    with TestClient(make_app(tmp_path, later, **SETTINGS)) as api:
        restored = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        assert restored == saved
        review = api.post(
            f"/sessions/{sid}/configuration-review",
            json={"user_id": "dev", "provider": "scripted", "message": "Inspect again"},
        )
        assert review.status_code == 200, review.text
        assert review.json()["configuration"]["status"] == "matching", review.text
        assert later.calls == []
        request = {**body(saved), "inspection_only": False}
        run = submit_and_finish(api, sid, request)
        assert run["response"]["completion_status"] == "blocked", run
        fork_id = run["request"]["session_id"]
        fork = api.get(f"/sessions/{fork_id}", params={"user_id": "dev"}).json()
        assert fork["session"]["inspection_only"] is True
        assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == saved
    assert len(later.calls) == 1
    assert "write_file" not in {spec.name for spec in later.calls[0][1]}
    assert target.read_text() == "user = 1\n"


@pytest.mark.parametrize("transport", ["http", "stream"])
def test_inspection_block_is_durable_and_keyed_replay_dispatches_nothing(tmp_path, transport):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "user.py"
    target.write_text("user = 1\n")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "user.py", "content": "overwritten"},
                )
            ]
        )
    )
    with TestClient(make_app(tmp_path, provider)) as api:
        sid = api.post(
            "/sessions",
            json={"user_id": "dev", "workspace_root": str(root), "inspection_only": True},
        ).json()["session_id"]
        payload = {
            "user_id": "dev",
            "session_id": sid,
            "message": "Edit this file",
            "request_id": "inspection-block",
        }

        def submit():
            if transport == "http":
                result = api.post("/chat", json=payload)
                assert result.status_code == 200, result.text
                return result.json()
            result = api.get("/chat/stream", params=payload)
            assert result.status_code == 200, result.text
            events = _parse_sse(result.text)
            assert not any(e["event"] in {"tool_start", "tool_end"} for e in events)
            return json.loads(next(e["data"] for e in events if e["event"] == "turn_done"))[
                "response"
            ]

        first = submit()
        assert first["completion_status"] == "blocked" and first["tool_calls"] == []
        assert first["verification_status"] == "not_run"
        assert submit() == first
        assert len(provider.calls) == 1
        saved = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        assert saved["responses"][0]["response"]["completion_status"] == "blocked"
        assert saved["responses"][0]["execution_profile"]["inspection_only"] is True
    assert target.read_text() == "user = 1\n"
