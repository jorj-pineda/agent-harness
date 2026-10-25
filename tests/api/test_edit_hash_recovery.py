"""Scripted correction mechanics with real tools/storage, never model-quality evidence."""

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from providers.base import ToolCall

from .conftest import ScriptedProvider, make_response
from .test_session_persistence import make_app
from .test_stream import _parse_sse


@pytest.mark.parametrize("transport", ["http", "stream"])
@pytest.mark.parametrize("budget_stop", [False, True])
def test_hash_correction_and_token_prevention_survive_restart(tmp_path, transport, budget_stop):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = b"user = 1\nagent = 0\n"
    target = repo / "a.py"
    target.write_bytes(source)
    target.chmod(0o640)
    protected = repo / "user.txt"
    protected.write_bytes(b"pre-existing user work\n")
    protected.chmod(0o600)
    visible = repo / "test_visible.py"
    visible_source = (
        "from a import user, agent\n\ndef test_values():\n    assert (user, agent) == (1, 2)\n"
    )
    visible.write_text(visible_source)
    digest = hashlib.sha256(source).hexdigest()
    arguments = {
        "path": "a.py",
        "old_text": "agent = 0",
        "new_text": "agent = 2",
        "expected_sha256": digest,
    }
    check = ["python", "-m", "pytest", "-q", "test_visible.py"]
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[ToolCall(id="read", name="read_file", arguments={"path": "a.py"})],
            prompt_tokens=100,
            completion_tokens=5,
        ),
        make_response(
            tool_calls=[
                ToolCall(
                    id="invalid-hash",
                    name="replace_text",
                    arguments={**arguments, "expected_sha256": digest[:-1]},
                )
            ],
            prompt_tokens=100,
            completion_tokens=5,
        ),
        make_response(
            tool_calls=[ToolCall(id="corrected-hash", name="replace_text", arguments=arguments)],
            # Scripted over-reporting exercises the existing post-response gate.
            # This is not a live usage claim or permission to overshoot a budget.
            prompt_tokens=64000 if budget_stop else 100,
            completion_tokens=5,
        ),
        make_response(
            tool_calls=[ToolCall(id="check", name="run_command", arguments={"argv": check})],
            prompt_tokens=100,
            completion_tokens=5,
        ),
        make_response(
            content="Changed a.py; configured check passed.", prompt_tokens=100, completion_tokens=5
        ),
    )
    settings = {
        "max_total_tokens_per_turn": 64000,
        "project_check_argv": check,
        "require_verification_before_finish": True,
    }
    with TestClient(make_app(tmp_path, provider, **settings)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        payload = {
            "user_id": "dev",
            "session_id": sid,
            "message": "Change only agent to 2, preserving user work, and check.",
            "request_id": "hash-correction",
        }
        if transport == "http":
            response = client.post("/chat", json=payload)
            assert response.status_code == 200, response.text
            result = response.json()
        else:
            response = client.get("/chat/stream", params=payload)
            assert response.status_code == 200, response.text
            result = json.loads(_parse_sse(response.text)[-1]["data"])["response"]
        assert len(provider.calls) == (3 if budget_stop else 5)
        recovery_messages = provider.calls[2][0]
        read_result = next(m for m in recovery_messages if m.tool_call_id == "read")
        assert json.loads(read_result.content)["sha256"] == digest
        feedback = next(m for m in recovery_messages if m.tool_call_id == "invalid-hash")
        assert "Copy the complete sha256 from the latest read_file result" in feedback.content
        assert "arguments were not rewritten" in feedback.content
        edits = [t for t in result["tool_calls"] if t["name"] == "replace_text"]
        assert len(edits) == 2
        assert edits[0]["arguments"]["expected_sha256"] == digest[:-1]
        assert "exactly 64 lowercase hexadecimal" in edits[0]["error"]
        assert edits[1]["arguments"] == arguments
        if budget_stop:
            assert result["completion_status"] == "budget_exhausted"
            assert result["verification_status"] == "not_run"
            assert edits[1]["error"] == "Tool not executed: this turn has already stopped."
            assert target.read_bytes() == source
            assert result["token_usage"] == {"prompt_tokens": 64200, "completion_tokens": 15}
        else:
            assert result["completion_status"] == "completed"
            assert result["verification_status"] == "passed"
            assert edits[1]["error"] is None
            assert edits[1]["result"]["replacement_count"] == 1
            assert target.read_bytes() == b"user = 1\nagent = 2\n"
        assert target.stat().st_mode & 0o777 == 0o640
        endpoint = f"/runs/{result['run_id']}"
        saved = client.get(endpoint, params={"user_id": "dev"}).json()
        assert saved["response"] == result
        if budget_stop:
            assert saved["budget_evidence"]["events"][-1]["kind"] == "post_response_stop"
        assert len(client.get("/runs", params={"user_id": "dev"}).json()) == 1
    # Saved review/keyed replay must preserve later work rather than repeat tools.
    later = target.read_bytes() + b"later_user = True\n"
    target.write_bytes(later)
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted, **settings)) as client:
        assert client.get(endpoint, params={"user_id": "dev"}).json() == saved
        assert client.post("/chat", json=payload).json() == result
        assert restarted.calls == []
        assert len(client.get("/runs", params={"user_id": "dev"}).json()) == 1
        assert target.read_bytes() == later
        assert target.stat().st_mode & 0o777 == 0o640
        assert protected.read_bytes() == b"pre-existing user work\n"
        assert protected.stat().st_mode & 0o777 == 0o600
        assert visible.read_text() == visible_source
