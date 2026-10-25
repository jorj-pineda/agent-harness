"""Real-file API continuation/restart contracts with synthetic SDK HTTP only."""

import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from api.server import create_app
from harness.config import Settings
from providers import create_chat_provider
from tests.api.test_continuation import async_wait
from tests.api.test_continuation import body as continuation_body
from tests.api.test_stream import _parse_sse
from tests.test_compatible_continuation import reply


def tool_reply(text, name, arguments, call_id):
    data = reply(text).json()
    data["choices"][0]["finish_reason"] = "tool_calls"
    data["choices"][0]["message"]["content"] = None
    data["choices"][0]["message"]["tool_calls"] = [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
    ]
    return httpx.Response(200, json=data)


def settings_for(tmp_path, enabled=True):
    return Settings(
        _env_file=None,
        model_profiles={
            "scripted": {
                "provider": "openai_compatible",
                "model": "SCRIPTED",
                "max_context_tokens": 16384,
                "max_request_output_tokens": 2048,
            }
        },
        model_profile="scripted",
        default_provider="openai_compatible",
        openai_compatible_base_url="https://fixture.invalid/inference/go/openai/v1",
        openai_compatible_api_key="synthetic-private-key",
        openai_compatible_model="SCRIPTED",
        openai_compatible_transport="opencode_go",
        openai_compatible_replay_reasoning_content=enabled,
        enable_support_tools=False,
        memory_db_path=tmp_path / "facts.db",
        session_db_path=tmp_path / "sessions.db",
        max_context_tokens=16384,
        max_request_output_tokens=2048,
        project_check_argv=["pytest", "-q"],
        require_verification_before_finish=True,
    )


@pytest.mark.parametrize("transport", ["http", "sse"])
@pytest.mark.parametrize("guard", ["matching", "disabled", "workspace"])
def test_restart_preserves_exact_state_and_fork_enforces_configuration_and_workspace(
    tmp_path, transport, guard
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "work.py"
    target.write_text("value = 1\n")
    target.chmod(0o640)
    (workspace / "test_work.py").write_text(
        "from work import value\ndef test_value():\n    assert value == 2\n"
    )
    (workspace / "preexisting.txt").write_text("preserve\n")
    seen = []
    owners = []
    restart = False
    states = ["synthetic read 🙂\r\n", "synthetic write", "synthetic check", "synthetic final"]

    def respond(request):
        seen.append(request)
        wire = json.loads(request.content)
        if restart:
            assert guard == "matching"
            assert [
                m["reasoning_content"] for m in wire["messages"] if "reasoning_content" in m
            ] == states
            return reply("synthetic fork state")
        number = len(seen)
        previous = [m["reasoning_content"] for m in wire["messages"] if "reasoning_content" in m]
        assert previous == states[: number - 1]
        if number == 1:
            return tool_reply(states[0], "read_file", {"path": "work.py"}, "read")
        if number == 2:
            return tool_reply(
                states[1], "write_file", {"path": "work.py", "content": "value = 2\n"}, "write"
            )
        if number == 3:
            return tool_reply(states[2], "run_command", {"argv": ["pytest", "-q"]}, "check")
        assert number == 4
        return reply(states[3])

    def construct(kind, **kwargs):
        assert kind == "openai" and kwargs["provider_name"] == "openai_compatible"
        adapter = create_chat_provider(
            kind, **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
        )
        owners.append(adapter)
        return adapter

    with patch("harness.providers.create_chat_provider", side_effect=construct):
        with TestClient(create_app(settings=settings_for(tmp_path))) as api:
            sid = api.post(
                "/sessions",
                json={"user_id": "dev", "workspace_root": str(workspace), "checkpoint_turns": True},
            ).json()["session_id"]
            payload = {
                "user_id": "dev",
                "session_id": sid,
                "message": "Update work.py and run the configured check",
                "request_id": "source",
            }
            if transport == "http":
                response = api.post("/chat", json=payload)
                assert response.status_code == 200, response.text
                final = response.json()
            else:
                response = api.get("/chat/stream", params=payload)
                assert response.status_code == 200, response.text
                final = json.loads(_parse_sse(response.text)[-1]["data"])["response"]
            assert (
                final["completion_status"] == "completed"
                and final["verification_status"] == "passed"
            )
            assert not any(s in final["answer"] for s in states)
            archived = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
            saved = api.get(f"/runs/{final['run_id']}", params={"user_id": "dev"}).json()
            messages = archived["session"]["messages"]
            assert [m["continuation"]["text"] for m in messages if "continuation" in m] == states
        assert owners[0]._client.is_closed() and len(seen) == 4
        assert all(r.headers["x-opencode-session"] == sid for r in seen)
        if guard == "workspace":
            target.write_text("value = 2\nlater_user = 3\n")
        before = target.read_bytes()
        restart = True
        with TestClient(
            create_app(settings=settings_for(tmp_path, enabled=guard != "disabled"))
        ) as api:
            assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archived
            assert api.get(f"/runs/{saved['run_id']}", params={"user_id": "dev"}).json() == saved
            assert api.get(f"/sessions/{sid}", params={"user_id": "other"}).status_code == 403
            assert len(seen) == 4
            # Restored history remains read-only; it cannot directly dispatch.
            assert api.post("/chat", json={**payload, "request_id": "different"}).status_code == 409
            fork_request = continuation_body(archived)
            fork_request["message"] = "Report the saved work.py change"
            response = api.post(f"/sessions/{sid}/continue", json=fork_request)
            assert response.status_code == 202, response.text
            run_id = response.json()["run_id"]
            api.portal.call(async_wait, api.app.state.components, run_id)
            fork = api.get(f"/runs/{run_id}", params={"user_id": "dev"}).json()
            if guard == "matching":
                assert fork["response"]["completion_status"] == "completed", fork
                assert len(seen) == 5
                assert (
                    seen[-1].headers["x-opencode-session"] == fork["request"]["session_id"] != sid
                )
            else:
                assert fork["response"]["completion_status"] == "blocked", fork
                assert len(seen) == 4 and fork["before_checkpoint_id"] is None
            assert api.post(f"/sessions/{sid}/continue", json=fork_request).json() == fork
            assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archived
            assert api.get(f"/runs/{saved['run_id']}", params={"user_id": "dev"}).json() == saved
        assert all(p._client.is_closed() for p in owners)
    assert target.read_bytes() == before and target.stat().st_mode & 0o777 == 0o640
    assert (workspace / "preexisting.txt").read_text() == "preserve\n"


@pytest.mark.parametrize("transport", ["http", "sse"])
@pytest.mark.parametrize(
    "invalid", [{"private-failure-marker": "not text"}, "x" * 65537], ids=["nontext", "oversize"]
)
def test_invalid_state_rejects_before_edit_and_restart_retains_failed_evidence(
    tmp_path, transport, invalid
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "work.py"
    target.write_text("preexisting_user = 1\n")
    target.chmod(0o640)
    seen = []
    owners = []

    def respond(request):
        seen.append(request)
        return tool_reply(
            invalid,
            "write_file",
            {"path": "work.py", "content": "must not apply\n"},
            "blocked-write",
        )

    def construct(kind, **kwargs):
        assert kind == "openai" and kwargs["provider_name"] == "openai_compatible"
        adapter = create_chat_provider(
            kind, **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
        )
        owners.append(adapter)
        return adapter

    with patch("harness.providers.create_chat_provider", side_effect=construct):
        with TestClient(
            create_app(settings=settings_for(tmp_path)), raise_server_exceptions=False
        ) as api:
            sid = api.post(
                "/sessions",
                json={"user_id": "dev", "workspace_root": str(workspace), "checkpoint_turns": True},
            ).json()["session_id"]
            payload = {
                "user_id": "dev",
                "session_id": sid,
                "message": "Bounded edit",
                "request_id": "failed-once",
            }
            response = (
                api.post("/chat", json=payload)
                if transport == "http"
                else api.get("/chat/stream", params=payload)
            )
            assert response.status_code == (500 if transport == "http" else 200)
            assert "private-failure-marker" not in response.text
            if transport == "sse":
                assert _parse_sse(response.text)[-1]["event"] == "error"
            saved = api.get("/runs", params={"user_id": "dev"}).json()
            assert len(saved) == 1
            saved = saved[0]
            assert saved["status"] == "failed" and saved["response"] is None
            assert (
                saved["before_checkpoint_id"] is not None and saved["after_checkpoint_id"] is None
            )
            evidence = saved["budget_evidence"]["events"]
            assert [e["kind"] for e in evidence] == ["request", "request_failure"]
            assert evidence[-1]["usage"] is None
            archived = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
            assert archived["responses"] == [] and archived["session"]["turns"] == []
            assert all(m["role"] != "assistant" for m in archived["session"]["messages"])
        assert owners[0]._client.is_closed() and len(seen) == 1
        target.write_text(target.read_text() + "later_user = 2\n")
        with TestClient(create_app(settings=settings_for(tmp_path))) as api:
            assert api.get(f"/runs/{saved['run_id']}", params={"user_id": "dev"}).json() == saved
            assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archived
            assert len(seen) == 1
        assert all(p._client.is_closed() for p in owners)
    assert target.read_text() == "preexisting_user = 1\nlater_user = 2\n"
    assert target.stat().st_mode & 0o777 == 0o640
