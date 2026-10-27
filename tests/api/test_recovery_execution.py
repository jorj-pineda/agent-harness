"""One explicit projected recovery turn with real tools, fake inference and durable review."""

import json
import stat
import subprocess

import pytest
from fastapi.testclient import TestClient

from harness.recovery_identity import recovery_messages_sha256
from providers.base import ChatMessage, ProviderError, ToolCall

from .conftest import ScriptedProvider, make_response
from .test_continuation import async_wait
from .test_session_persistence import make_app
from .test_stream import _parse_sse

SETTINGS = {
    "max_context_tokens": 16384,
    "ollama_num_ctx": 16384,
    "ollama_model": "gemma4:12b",
    "coding_prompt_profile": "compact",
    "max_tool_iterations": 8,
    "max_request_output_tokens": 1024,
    "max_completion_tokens_per_turn": 8192,
    "max_total_tokens_per_turn": 64000,
    "max_turn_wall_seconds": 300,
    "max_completion_retries": 0,
    "require_verification_before_finish": True,
    "project_check_argv": ["pytest", "-q"],
}
PARTIAL = "user = 1\nagent = 2\n"
RECOVERED = PARTIAL + "recovered = 3\n"


def observed_response(**kwargs):
    return make_response(prompt_tokens=128, completion_tokens=32, **kwargs)


def known_provider():
    provider = ScriptedProvider()
    provider.name = "ollama"
    return provider


def source(tmp_path, *, transport="http", inspection=False, settings=None):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("user = 1\n")
    (root / "a.py").chmod(0o640)
    (root / "test_a.py").write_text(
        "import a\n\ndef test_result():\n    assert (a.user, a.agent, a.recovered) == (1, 2, 3)\n"
    )
    (root / "user.txt").write_text("pre-existing staged user work\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "user.txt"], check=True)
    before_index = (root / ".git/index").read_bytes()
    provider = known_provider()
    if inspection:
        provider.script(
            observed_response(
                tool_calls=[ToolCall(id="read", name="read_file", arguments={"path": "a.py"})]
            ),
            observed_response(content="Read-only source result"),
        )
    else:
        provider.script(
            observed_response(
                tool_calls=[
                    ToolCall(
                        id="partial",
                        name="write_file",
                        arguments={"path": "a.py", "content": PARTIAL},
                    )
                ]
            ),
            observed_response(content=""),
        )
    options = settings or SETTINGS
    with TestClient(make_app(tmp_path, provider, **options)) as client:
        sid = client.post(
            "/sessions",
            json={
                "user_id": "dev",
                "workspace_root": str(root),
                "checkpoint_turns": True,
                "inspection_only": inspection,
            },
        ).json()["session_id"]
        payload = {
            "user_id": "dev",
            "session_id": sid,
            "message": "Inspect a.py" if inspection else "Update a.py",
        }
        if transport == "http":
            response = client.post("/chat", json=payload).json()
        else:
            response = json.loads(
                _parse_sse(client.get("/chat/stream", params=payload).text)[-1]["data"]
            )["response"]
        rid = response["run_id"]
        assert response["completion_status"] == ("completed" if inspection else "incomplete")
        record = client.get(f"/runs/{rid}", params={"user_id": "dev"}).json()
        archive = client.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
        checkpoint = client.get(
            f"/sessions/{sid}/checkpoints/{record['after_checkpoint_id']}",
            params={"user_id": "dev"},
        ).json()
        saved_hash = client.get(f"/runs/{rid}/recovery-review", params={"user_id": "dev"}).json()[
            "source_record_sha256"
        ]
        body = {
            "user_id": "dev",
            "expected_record_sha256": saved_hash,
            "spans": [
                {
                    "path": "a.py",
                    "expected_sha256": checkpoint["files"]["a.py"]["sha256"],
                    "start_line": 1,
                    "end_line": 1 if inspection else 2,
                }
            ],
            "message": "Explain a.py"
            if inspection
            else "Complete a.py with recovered = 3, run the configured check, then report.",
        }
        draft_result = client.post(f"/runs/{rid}/recovery-draft", json=body)
        assert draft_result.status_code == 200, draft_result.text
        draft = draft_result.json()
        assert draft["configuration"]["status"] == "matching"
        approval = {
            **body,
            "request_id": "recover-once",
            "expected_archive_sha256": draft["source_archive_sha256"],
            "expected_messages_sha256": draft["messages_sha256"],
            "expected_tool_specs_sha256": draft["tool_specs_sha256"],
            "expected_request_output_allowance": draft["context"]["request_output_allowance"],
            "acknowledge_uncaptured_state": True,
            "acknowledge_omitted_context": True,
            "acknowledge_new_turn_budget": True,
        }
    return {
        "root": root,
        "sid": sid,
        "rid": rid,
        "record": record,
        "archive": archive,
        "draft": draft,
        "approval": approval,
        "index": before_index,
        "settings": options,
    }


def recover(client, data, *, failure=None):
    result = client.post(f"/runs/{data['rid']}/recover", json=data["approval"])
    assert result.status_code == 202, result.text
    rid = result.json()["run_id"]
    if failure is None:
        client.portal.call(async_wait, client.app.state.components, rid)
    else:
        with pytest.raises(failure):
            client.portal.call(async_wait, client.app.state.components, rid)
    return client.get(f"/runs/{rid}", params={"user_id": "dev"}).json()


@pytest.mark.parametrize("transport", ["http", "stream"])
def test_projected_recovery_executes_exact_input_checks_and_preserves_source_after_restart(
    tmp_path, transport
):
    data = source(tmp_path, transport=transport)
    provider = known_provider()
    provider.script(
        observed_response(
            tool_calls=[
                ToolCall(
                    id="recover",
                    name="write_file",
                    arguments={"path": "a.py", "content": RECOVERED},
                )
            ]
        ),
        observed_response(
            tool_calls=[
                ToolCall(id="check", name="run_command", arguments={"argv": ["pytest", "-q"]})
            ]
        ),
        observed_response(content="Changed a.py and the configured check passed."),
    )
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:
        run = recover(client, data)
        assert run["status"] == "finished", run
        assert run["response"]["completion_status"] == "completed", run
        assert run["response"]["verification_status"] == "passed"
        assert run["before_checkpoint_id"] and run["after_checkpoint_id"]
        assert (
            run["execution_profile"]["prompt_version"] is None
        )  # Explicit custom recovery framing.
        provenance = run["request"]["recovery_source"]
        assert provenance["messages_sha256"] == data["draft"]["messages_sha256"]
        assert provenance["budget_scope"] == "separate-configured-turn"
        first = [ChatMessage.model_validate(m) for m in data["draft"]["messages"]]
        assert provider.calls[0][0] == first
        assert recovery_messages_sha256(provider.calls[0][0]) == provenance["messages_sha256"]
        assert len(provider.calls) == 3
        events = run["budget_evidence"]["events"]
        first_request = next(e for e in events if e["kind"] == "request")
        assert first_request["max_tokens"] == data["draft"]["context"]["request_output_allowance"]
        fork = run["request"]["session_id"]
        fork_archive = client.get(f"/sessions/{fork}", params={"user_id": "dev"}).json()
        assert len(fork_archive["responses"]) == 2
        assert fork_archive["session"]["recovery_source"] == provenance
        assert fork not in client.app.state.components.sessions
        assert (
            client.post(
                "/chat", json={"user_id": "dev", "session_id": fork, "message": "Bypass"}
            ).status_code
            == 409
        )
        assert (
            client.post(
                f"/sessions/{fork}/continue",
                json={
                    "user_id": "dev",
                    "request_id": "bypass",
                    "expected_turn_id": fork_archive["session"]["turns"][-1]["turn_id"],
                    "message": "Bypass",
                    "acknowledge_uncaptured_state": True,
                },
            ).status_code
            == 409
        )
        assert (
            client.get(f"/sessions/{data['sid']}", params={"user_id": "dev"}).json()
            == data["archive"]
        )
        assert (
            client.get(f"/runs/{data['rid']}", params={"user_id": "dev"}).json() == data["record"]
        )
    assert (data["root"] / "a.py").read_text() == RECOVERED
    assert stat.S_IMODE((data["root"] / "a.py").stat().st_mode) == 0o640
    assert (data["root"] / ".git/index").read_bytes() == data["index"]
    assert (data["root"] / "user.txt").read_text() == "pre-existing staged user work\n"
    (data["root"] / "a.py").write_text(RECOVERED + "# later user work\n")
    restarted = known_provider()
    with TestClient(make_app(tmp_path, restarted, **SETTINGS)) as client:
        replay = client.post(f"/runs/{data['rid']}/recover", json=data["approval"])
        assert replay.status_code == 202 and replay.json() == run
        assert client.get(f"/sessions/{fork}", params={"user_id": "dev"}).json() == fork_archive
        assert (
            client.post(
                f"/runs/{data['rid']}/recover", json={**data["approval"], "message": "different"}
            ).status_code
            == 409
        )
        assert len(client.get("/runs", params={"user_id": "dev"}).json()) == 2
    assert restarted.calls == []
    assert (data["root"] / "a.py").read_text().endswith("# later user work\n")


@pytest.mark.parametrize(
    "patch, status",
    [
        ({"acknowledge_new_turn_budget": False}, 422),
        ({"acknowledge_omitted_context": 1}, 422),
        ({"acknowledge_uncaptured_state": "true"}, 422),
        ({"expected_messages_sha256": "0" * 64}, 409),
        ({"expected_archive_sha256": "0" * 64}, 409),
        ({"expected_tool_specs_sha256": "0" * 64}, 409),
        ({"expected_request_output_allowance": 1}, 409),
        ({"user_id": "other"}, 403),
        ({"inspection_only": False}, 422),
    ],
)
def test_invalid_approval_creates_no_run_or_fork(tmp_path, patch, status):
    data = source(tmp_path)
    provider = known_provider()
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:
        result = client.post(f"/runs/{data['rid']}/recover", json={**data["approval"], **patch})
        assert result.status_code == status, result.text
        assert len(client.get("/runs", params={"user_id": "dev"}).json()) == 1
        assert client.app.state.components.sessions == {}
    assert provider.calls == []
    assert (data["root"] / "a.py").read_text() == PARTIAL


@pytest.mark.parametrize("change", ["workspace", "mode", "instructions", "facts", "configuration"])
def test_changes_after_seed_refuse_before_checkpoint_or_provider(tmp_path, monkeypatch, change):
    import api.server as server

    data = source(tmp_path)
    original = server._start_run

    def seed_then_change(*args, **kwargs):
        result = original(*args, **kwargs)
        if change == "workspace":
            (data["root"] / "a.py").write_text(PARTIAL + "# later user edit\n")
        elif change == "mode":
            (data["root"] / "a.py").chmod(0o600)
        elif change == "instructions":
            (data["root"] / "AGENTS.md").write_text("New user instructions\n")
        elif change == "facts":
            args[0].fact_store.add("dev", "Preserve the new user constraint")
        else:
            args[1].ollama_model = "changed-model"
        return result

    monkeypatch.setattr(server, "_start_run", seed_then_change)
    provider = known_provider()
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:
        run = recover(client, data)
        assert run["response"]["completion_status"] == "blocked", run
        assert {"facts": "draft_changed", "configuration": "configuration_unusable"}.get(
            change, "workspace_changed"
        ) in run["response"]["completion_reason"]
        assert run["before_checkpoint_id"] is None and run["after_checkpoint_id"] is None
        assert client.app.state.components.sessions == {}
        assert (
            client.get(f"/sessions/{data['sid']}", params={"user_id": "dev"}).json()
            == data["archive"]
        )
    assert provider.calls == []
    if change == "workspace":
        assert (data["root"] / "a.py").read_text().endswith("# later user edit\n")


def test_actual_first_input_drift_never_reaches_provider(tmp_path, monkeypatch):
    import harness.runtime as runtime

    data = source(tmp_path)
    original = runtime.refresh_system_message

    def drift(session, *args, **kwargs):
        result = original(session, *args, **kwargs)
        if kwargs.get("system_suffix"):
            session.messages[0].content += "\nUnexpected drift"
        return result

    monkeypatch.setattr(runtime, "refresh_system_message", drift)
    provider = known_provider()
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:
        run = recover(client, data, failure=ProviderError)
        assert run["status"] == "failed", run
        assert run["response"] is None
        assert client.app.state.components.sessions == {}
    assert provider.calls == []
    assert (data["root"] / "a.py").read_text() == PARTIAL


def test_cancel_after_edit_retains_partial_review_and_replay_without_redispatch(
    tmp_path, monkeypatch
):
    import asyncio
    import threading

    data = source(tmp_path)
    provider = known_provider()
    provider.script(
        observed_response(
            tool_calls=[
                ToolCall(
                    id="recover",
                    name="write_file",
                    arguments={"path": "a.py", "content": RECOVERED},
                )
            ]
        )
    )
    original = provider.chat
    started = threading.Event()

    async def hold(*args, **kwargs):
        if provider.calls:
            started.set()
            await asyncio.Event().wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(provider, "chat", hold)
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:
        result = client.post(f"/runs/{data['rid']}/recover", json=data["approval"])
        assert result.status_code == 202, result.text
        rid = result.json()["run_id"]
        fork = result.json()["request"]["session_id"]
        assert started.wait(3)
        try:
            payload = {"user_id": "dev", "session_id": fork, "message": "Bypass"}
            assert client.post("/chat", json=payload).status_code == 409
            assert client.get("/chat/stream", params=payload).status_code == 409
            assert client.post("/runs", json={**payload, "request_id": "bypass"}).status_code == 409
        finally:
            cancelled = client.post(f"/sessions/{fork}/cancel", json={"user_id": "dev"})
            assert cancelled.status_code == 202, cancelled.text
            client.portal.call(async_wait, client.app.state.components, rid)
        run = client.get(f"/runs/{rid}", params={"user_id": "dev"}).json()
        assert run["response"]["completion_status"] == "cancelled", run
        assert run["response"]["token_usage"]["completion_tokens"] is None
        assert run["before_checkpoint_id"] and run["after_checkpoint_id"]
        assert "+recovered = 3" in run["response"]["workspace_changes"]["diffs"][0]["diff"]
        assert client.app.state.components.sessions == {}
        assert (
            client.get(f"/sessions/{data['sid']}", params={"user_id": "dev"}).json()
            == data["archive"]
        )
    restarted = known_provider()
    with TestClient(make_app(tmp_path, restarted, **SETTINGS)) as client:
        replay = client.post(f"/runs/{data['rid']}/recover", json=data["approval"])
        assert replay.json() == run
    assert restarted.calls == []
    assert (data["root"] / "a.py").read_text() == RECOVERED


def test_inspection_policy_survives_projection_and_rejects_write_batch(tmp_path):
    data = source(tmp_path, inspection=True)
    provider = known_provider()
    provider.script(
        observed_response(
            tool_calls=[
                ToolCall(id="read", name="read_file", arguments={"path": "a.py"}),
                ToolCall(id="write", name="write_file", arguments={"path": "a.py", "content": "X"}),
            ]
        ),
        observed_response(content="Inspection only."),
    )
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:
        run = recover(client, data)
        assert run["status"] == "finished", run
        archive = client.get(
            f"/sessions/{run['request']['session_id']}", params={"user_id": "dev"}
        ).json()
        assert archive["session"]["inspection_only"] is True
        assert run["response"]["workspace_changes"]["modified"] == []
    assert (data["root"] / "a.py").read_text() == "user = 1\n"


@pytest.mark.parametrize("failure", ["provider", "final_storage"])
def test_failed_execution_has_no_invented_final_and_restart_replay_is_read_only(
    tmp_path, monkeypatch, failure
):
    import sqlite3

    data = source(tmp_path)
    provider = known_provider()
    provider.script(
        observed_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={
                        "path": "a.py",
                        "content": RECOVERED,
                    },
                )
            ]
        ),
        observed_response(content="Edited."),
    )
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:
        if failure == "provider":

            async def fail(*args, **kwargs):
                raise ProviderError("Synthetic provider failure")

            monkeypatch.setattr(provider, "chat", fail)
        else:

            def fail_storage(*args, **kwargs):
                raise sqlite3.OperationalError("Synthetic final storage failure")

            monkeypatch.setattr(
                client.app.state.components.session_store, "finish_run", fail_storage
            )
        run = recover(client, data, failure=Exception)
        assert run["status"] == "failed", run
        assert run["response"] is None
        assert run["before_checkpoint_id"]
        assert bool(run["after_checkpoint_id"]) == (failure == "final_storage")
        assert client.app.state.components.sessions == {}
        assert (
            client.get(f"/sessions/{data['sid']}", params={"user_id": "dev"}).json()
            == data["archive"]
        )
    with TestClient(make_app(tmp_path, known_provider(), **SETTINGS)) as client:
        replay = client.post(f"/runs/{data['rid']}/recover", json=data["approval"])
        assert replay.status_code == 202 and replay.json() == run
    assert (data["root"] / "a.py").read_text() == (
        RECOVERED if failure == "final_storage" else PARTIAL
    )


def test_busy_preparation_refuses_without_creating_intent(tmp_path):
    from harness.admission import turn_admission

    data = source(tmp_path)
    provider = known_provider()
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:
        with turn_admission.claim("other-session", str(data["root"])) as busy:
            assert busy is None
            response = client.post(f"/runs/{data['rid']}/recover", json=data["approval"])
        assert response.status_code == 409, response.text
        assert len(client.get("/runs", params={"user_id": "dev"}).json()) == 1
        assert client.app.state.components.sessions == {}
    assert provider.calls == []


def test_missing_new_turn_usage_stops_before_tool_dispatch(tmp_path):
    data = source(tmp_path)
    provider = known_provider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="unknown-usage",
                    name="write_file",
                    arguments={
                        "path": "a.py",
                        "content": RECOVERED,
                    },
                )
            ]
        )
    )
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as client:
        run = recover(client, data)
        assert run["response"]["completion_status"] == "incomplete", run
        assert "did not report output-token usage" in run["response"]["completion_reason"]
        assert run["response"]["token_usage"]["completion_tokens"] is None
        assert run["before_checkpoint_id"] and run["after_checkpoint_id"]
    assert len(provider.calls) == 1
    assert (data["root"] / "a.py").read_text() == PARTIAL
