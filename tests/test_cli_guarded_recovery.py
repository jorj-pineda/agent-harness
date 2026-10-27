"""Explicit CLI submission preserves reviewed identity and never silently retries."""

import hashlib
import json
import uuid
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli.main import app
from harness.recovery_admission import RecoveryApproval
from providers.base import ToolCall
from tests.api.test_continuation import async_wait
from tests.api.test_recovery_execution import (
    RECOVERED,
    SETTINGS,
    known_provider,
    observed_response,
    source,
)
from tests.api.test_session_persistence import make_app

runner = CliRunner()
ACKS = {
    "acknowledge_uncaptured_state": True,
    "acknowledge_omitted_context": True,
    "acknowledge_new_turn_budget": True,
}


def ready(tmp_path):
    data = source(tmp_path)
    approval = {
        k: v for k, v in data["approval"].items() if k not in {*ACKS, "user_id", "request_id"}
    }
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(approval))
    return data, approval, path


def args(data, path):
    return [
        "recover-run",
        data["rid"],
        "--approval-file",
        str(path),
        "--request-id",
        "cli-once",
        "--user-id",
        "dev",
        "--acknowledge-uncaptured-state",
        "--acknowledge-omitted-context",
        "--acknowledge-new-turn-budget",
    ]


def record(data, approval):
    approved = RecoveryApproval(**approval, **ACKS).model_dump_json()
    fork = uuid.uuid5(
        uuid.NAMESPACE_URL,
        json.dumps(
            [
                "agent-harness-recovery",
                "dev",
                data["rid"],
                "cli-once",
            ]
        ),
    ).hex
    return {
        "run_id": "child-run",
        "status": "running",
        "response": None,
        "request": {
            "user_id": "dev",
            "request_id": "cli-once",
            "provider": None,
            "session_id": fork,
            "message": approval["message"],
            "continuation_source": {
                "session_id": data["sid"],
                "turn_id": data["archive"]["session"]["turns"][-1]["turn_id"],
                "acknowledge_uncaptured_state": True,
            },
            "recovery_source": {
                "schema_version": 1,
                "draft_version": "recovery-draft-v1",
                "source_run_id": data["rid"],
                "source_session_id": data["sid"],
                "source_turn_id": data["archive"]["session"]["turns"][-1]["turn_id"],
                "source_record_sha256": approval["expected_record_sha256"],
                "source_archive_sha256": approval["expected_archive_sha256"],
                "messages_sha256": approval["expected_messages_sha256"],
                "tool_specs_sha256": approval["expected_tool_specs_sha256"],
                "request_output_allowance": approval["expected_request_output_allowance"],
                "budget_scope": "separate-configured-turn",
                "approval_json": approved,
                "approval_sha256": hashlib.sha256(approved.encode()).hexdigest(),
            },
        },
    }


def response(value, status=202):
    return httpx.Response(status, json=value, request=httpx.Request("POST", "http://fixture"))


def test_human_and_json_make_one_post_validate_identity_and_never_fetch(tmp_path):
    data, approval, path = ready(tmp_path)
    saved = record(data, approval)
    before = path.read_bytes()
    for raw in (False, True):
        with patch("httpx.post", return_value=response(saved)) as post, patch("httpx.get") as get:
            result = runner.invoke(app, [*args(data, path), *(["--json"] if raw else [])])
        assert result.exit_code == 0, result.output
        post.assert_called_once_with(
            f"http://localhost:8000/runs/{data['rid']}/recover",
            json={"user_id": "dev", "request_id": "cli-once", "provider": None, **approval, **ACKS},
            timeout=30,
        )
        get.assert_not_called()
        if raw:
            assert json.loads(result.output) == saved
        else:
            assert 'completion="unknown" checks="unknown"' in result.output
            assert "does not prove completion or passed guards" in result.output
            assert "no source allowance transfer" in result.output
    assert path.read_bytes() == before


@pytest.mark.parametrize("flag", list(ACKS))
def test_all_acknowledgments_are_required_before_reading_file_or_network(flag):
    invocation = args({"rid": "r"}, "/does-not-exist")
    invocation.remove("--" + flag.replace("_", "-"))
    with patch("httpx.post") as post:
        result = runner.invoke(app, invocation)
    assert result.exit_code == 2
    assert "All three" in result.output
    post.assert_not_called()


@pytest.mark.parametrize(
    "bad",
    [
        "oversize",
        "duplicate",
        "nested_duplicate",
        "nonobject",
        "utf8",
        "directory",
        "symlink",
        "allowance_bool",
        "extra_ack",
        "message",
        "hash",
        "missing",
        "overlap",
    ],
)
def test_bad_approval_refuses_without_network(tmp_path, bad):
    data, approval, path = ready(tmp_path)
    if bad == "oversize":
        path.write_bytes(b" " * 65537)
    elif bad == "duplicate":
        path.write_text('{"spans": [], "spans": []}')
    elif bad == "nested_duplicate":
        path.write_text('{"spans": [{"path":"a", "path":"b"}]}')
    elif bad == "nonobject":
        path.write_text("[]")
    elif bad == "utf8":
        path.write_bytes(b"\xff")
    elif bad == "directory":
        path = tmp_path
    elif bad == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(path)
        path = link
    else:
        if bad == "allowance_bool":
            approval["expected_request_output_allowance"] = True
        elif bad == "extra_ack":
            approval.update(ACKS)
        elif bad == "message":
            approval["message"] = "é" * 2049
        elif bad == "hash":
            approval["expected_messages_sha256"] = "private-marker"
        elif bad == "missing":
            del approval["expected_archive_sha256"]
        else:
            approval["spans"] *= 2
        path.write_text(json.dumps(approval))
    with patch("httpx.post") as post, patch("httpx.get") as get:
        result = runner.invoke(app, args(data, path))
    assert result.exit_code == 1, result.output
    assert "private-marker" not in result.output
    post.assert_not_called()
    get.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        "owner",
        "key",
        "message",
        "provider",
        "fork",
        "run",
        "source",
        "version",
        "archive",
        "messages",
        "schema",
        "allowance",
        "scope",
        "approval",
        "approval_hash",
        "ack",
        "continuation",
        "response",
    ],
)
def test_mismatched_response_never_prints_success_or_retries(tmp_path, change):
    data, approval, path = ready(tmp_path)
    saved = record(data, approval)
    request = saved["request"]
    origin = request["recovery_source"]
    targets = {
        "owner": (request, "user_id", "other"),
        "key": (request, "request_id", "other"),
        "message": (request, "message", "other"),
        "provider": (request, "provider", "other"),
        "fork": (request, "session_id", "other"),
        "run": (saved, "run_id", data["rid"]),
        "source": (origin, "source_run_id", "other"),
        "version": (origin, "schema_version", True),
        "archive": (origin, "source_archive_sha256", "0" * 64),
        "messages": (origin, "messages_sha256", "0" * 64),
        "schema": (origin, "tool_specs_sha256", "0" * 64),
        "allowance": (origin, "request_output_allowance", True),
        "scope": (origin, "budget_scope", "hypothetical-new-turn"),
        "approval": (origin, "approval_json", "{}"),
        "approval_hash": (origin, "approval_sha256", "0" * 64),
        "ack": (request["continuation_source"], "acknowledge_uncaptured_state", 1),
        "continuation": (request["continuation_source"], "turn_id", "other"),
        "response": (saved, "response", []),
    }
    target, key, value = targets[change]
    target[key] = value
    with patch("httpx.post", return_value=response(saved)) as post, patch("httpx.get") as get:
        result = runner.invoke(app, [*args(data, path), "--json"])
    assert result.exit_code == 1, result.output
    assert "Recovery intent recorded" not in result.output
    post.assert_called_once()
    get.assert_not_called()


@pytest.mark.parametrize(
    "failure", [httpx.ReadTimeout("private-marker"), response({"detail": "private-marker"}, 409)]
)
def test_transport_failure_preserves_key_without_private_errors_or_retry(tmp_path, failure):
    data, _, path = ready(tmp_path)
    with (
        patch(
            "httpx.post",
            **(
                {"side_effect": failure}
                if isinstance(failure, Exception)
                else {"return_value": failure}
            ),
        ) as post,
        patch("httpx.get") as get,
    ):
        result = runner.invoke(app, args(data, path))
    assert result.exit_code == 1
    assert 'request_id="cli-once"' in result.output
    assert "private-marker" not in result.output and "No automatic retry" in result.output
    post.assert_called_once()
    get.assert_not_called()


def test_cli_real_tools_then_restart_replay_preserves_source_and_later_work(tmp_path):
    data, _, path = ready(tmp_path)
    provider = known_provider()
    provider.script(
        observed_response(
            tool_calls=[
                ToolCall(
                    id="edit", name="write_file", arguments={"path": "a.py", "content": RECOVERED}
                )
            ]
        ),
        observed_response(
            tool_calls=[
                ToolCall(id="check", name="run_command", arguments={"argv": ["pytest", "-q"]})
            ]
        ),
        observed_response(content="Changed and checked."),
    )
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as api:

        def post(url, **kwargs):
            return api.post(url.removeprefix("http://localhost:8000"), json=kwargs["json"])

        with patch("httpx.post", side_effect=post) as posted, patch("httpx.get") as get:
            result = runner.invoke(app, [*args(data, path), "--json"])
        assert result.exit_code == 0, result.output
        submitted = json.loads(result.output)
        posted.assert_called_once()
        get.assert_not_called()
        api.portal.call(async_wait, api.app.state.components, submitted["run_id"])
        completed = api.get(f"/runs/{submitted['run_id']}", params={"user_id": "dev"}).json()
        assert completed["response"]["completion_status"] == "completed"
        assert completed["response"]["verification_status"] == "passed"
        assert (
            api.get(f"/sessions/{data['sid']}", params={"user_id": "dev"}).json() == data["archive"]
        )
    (data["root"] / "a.py").write_text(RECOVERED + "# later user edit\n")
    restarted = known_provider()
    with TestClient(make_app(tmp_path, restarted, **SETTINGS)) as api:
        with patch("httpx.post", side_effect=post) as posted, patch("httpx.get") as get:
            result = runner.invoke(app, [*args(data, path), "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == completed
        posted.assert_called_once()
        get.assert_not_called()
    assert len(provider.calls) == 3 and restarted.calls == []
    assert (data["root"] / "a.py").read_text().endswith("# later user edit\n")
    assert (data["root"] / ".git/index").read_bytes() == data["index"]
