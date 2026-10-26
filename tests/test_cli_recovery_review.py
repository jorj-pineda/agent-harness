"""Thin owner-scoped GET with explicit partial/unknown evidence and terminal escaping."""

import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli.main import app
from harness.recovery_review import build_recovery_review
from harness.runs import RunInput, RunRecord
from harness.state import (
    CheckRecord,
    ToolCallRecord,
    TurnResponse,
    WorkspaceChangeReport,
    WorkspaceFileDiff,
)
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tests.api.test_session_persistence import make_app

runner = CliRunner()


def packet(final=True):
    record = RunRecord(
        run_id="r1",
        request=RunInput(user_id="dev", session_id="s", request_id="once", message="Fix\x1b[2J"),
        provider="scripted",
        status="finished" if final else "interrupted",
    )
    if final:
        record.response = TurnResponse(
            answer="",
            provider="scripted",
            latency_ms=0,
            completion_status="budget_exhausted",
            verification_status="stale",
            check_attempts=[
                CheckRecord(
                    argv=["pytest"],
                    status="failed",
                    relevant=True,
                    superseded_by_edit=True,
                    error="failure\x1b[2J",
                )
            ],
            tool_calls=[
                ToolCallRecord(name="read_file", result="private raw result") for _ in range(12)
            ],
            workspace_changes=WorkspaceChangeReport(
                status="tracked", diffs=[WorkspaceFileDiff(path="a.py", diff="x" * 5000)]
            ),
        )
    return build_recovery_review(record).model_dump(mode="json")


def response(value, status=200):
    return httpx.Response(status, json=value, request=httpx.Request("GET", "http://fixture"))


def test_cli_human_preserves_stops_failures_omissions_truncation_and_escapes():
    with patch("httpx.get", return_value=response(packet())) as get, patch("httpx.post") as post:
        result = runner.invoke(app, ["recovery-review", "r1", "--user-id", "dev"])
    assert result.exit_code == 0, result.output
    assert 'completion="budget_exhausted" checks="stale"' in result.output
    assert '"status": "failed"' in result.output and '"superseded_by_edit": true' in result.output
    assert '"outcome": "returned"' in result.output and "not proof of success" in result.output
    assert "action_count=12 selected=8 omitted=4" in result.output
    assert '"truncated": true' in result.output and '"source_bytes": 5000' in result.output
    assert "\\u001b" in result.output and "\x1b" not in result.output
    assert "private raw result" not in result.output
    assert "no command, inference or continuation" in result.output
    get.assert_called_once_with(
        "http://localhost:8000/runs/r1/recovery-review", params={"user_id": "dev"}, timeout=30
    )
    post.assert_not_called()


def test_missing_final_human_and_json_do_not_invent_work_or_completion():
    value = packet(False)
    with patch("httpx.get", return_value=response(value)) as get:
        human = runner.invoke(app, ["recovery-review", "r1"])
        raw = runner.invoke(app, ["recovery-review", "r1", "--json"])
    assert human.exit_code == raw.exit_code == 0
    assert "completion=unknown checks=unknown workspace=unknown" in human.output
    assert "action_count=unknown selected=0 omitted=unknown" in human.output
    assert "check_count=unknown" in human.output and "diff_count=unknown" in human.output
    assert json.loads(raw.output) == value
    assert get.call_count == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("run_id", "wrong"),
        ("run_status", "running"),
        ("action_count", -1),
        ("omitted_actions", True),
        ("diffs", [None]),
        ("objective", None),
        ("scope_note", None),
    ],
)
def test_bad_envelopes_fail_before_human_or_json_output(field, value):
    data = packet()
    data[field] = value
    data["completion_reason"] = {"text": "private-marker"}
    with patch("httpx.get", return_value=response(data)) as get:
        result = runner.invoke(app, ["recovery-review", "r1", "--json"])
    assert result.exit_code == 1 and "private-marker" not in result.output
    assert "No automatic retry" in result.output
    assert get.call_count == 1


@pytest.mark.parametrize(
    "failure", [httpx.ReadTimeout("private-marker"), ValueError("private-marker")]
)
def test_private_errors_never_retry_or_fall_back(failure):
    with patch("httpx.get", side_effect=failure) as get, patch("httpx.post") as post:
        result = runner.invoke(app, ["recovery-review", "r1"])
    assert result.exit_code == 1 and "private-marker" not in result.output
    assert get.call_count == 1
    post.assert_not_called()


@pytest.mark.parametrize("status", [403, 404, 409, 413])
def test_server_refusals_are_read_only_without_retry(status):
    with patch("httpx.get", return_value=response({"detail": "private-marker"}, status)) as get:
        result = runner.invoke(app, ["recovery-review", "r1"])
    assert result.exit_code == 1 and "private-marker" not in result.output
    assert get.call_count == 1


def test_oversized_body_and_missing_final_inconsistency_are_rejected():
    data = packet(False)
    data["diff_count"] = 0
    for value in [data, {**packet(), "extra": "x" * 32768}]:
        with patch("httpx.get", return_value=response(value)) as get:
            result = runner.invoke(app, ["recovery-review", "r1", "--json"])
        assert result.exit_code == 1
        assert get.call_count == 1


@pytest.mark.parametrize(
    "field,value", [("completion_status", "completed"), ("recent_actions", [{}])]
)
def test_missing_final_rejects_invented_evidence(field, value):
    data = packet(False)
    data[field] = value
    with patch("httpx.get", return_value=response(data)):
        result = runner.invoke(app, ["recovery-review", "r1", "--json"])
    assert result.exit_code == 1
    assert "completed" not in result.output


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        "zero_actions",
        "unknown_actions",
        "unknown_omissions",
        "wrong_diffs",
        "wrong_checks",
        "unknown_check_omissions",
        "short_selection",
    ],
)
def test_contradictory_counts_fail_before_either_output_mode(change, raw):
    data = packet()
    if change == "zero_actions":
        data.update(action_count=0, omitted_actions=0)
    elif change == "unknown_actions":
        data["action_count"] = None
    elif change == "unknown_omissions":
        data["omitted_actions"] = None
    elif change == "wrong_diffs":
        data["diff_count"] = 2
    elif change == "wrong_checks":
        data["unresolved_check_count"] = 0
    elif change == "unknown_check_omissions":
        data["omitted_unresolved_checks"] = None
    else:
        data["recent_actions"].pop()
        data["omitted_actions"] += 1  # Arithmetic matches, but v1 requires the full eight.
    data["objective"]["text"] = "private-marker"
    with patch("httpx.get", return_value=response(data)) as get, patch("httpx.post") as post:
        result = runner.invoke(app, ["recovery-review", "r1", *(["--json"] if raw else [])])
    assert result.exit_code == 1
    assert "private-marker" not in result.output
    assert "recovery_review run_id=" not in result.output
    assert get.call_count == 1
    post.assert_not_called()


def test_real_cli_get_after_lifecycle_restart_preserves_user_edits_and_evidence(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "a.py"
    target.write_text("user = 1\n")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="write",
                    name="write_file",
                    arguments={"path": "a.py", "content": "user = 1\nagent = 2\n"},
                )
            ]
        )
    )
    with TestClient(make_app(tmp_path, provider, max_tool_iterations=1)) as api:
        sid = api.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        rid = api.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Update a.py"}
        ).json()["run_id"]
        saved = api.get(f"/runs/{rid}", params={"user_id": "dev"}).json()
        archive = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
    target.write_text("later user edits\n")
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as api:

        def get(url, **kwargs):
            assert kwargs["timeout"] == 30
            return api.get(url.removeprefix("http://localhost:8000"), params=kwargs["params"])

        with patch("httpx.get", side_effect=get) as gets, patch("httpx.post") as post:
            args = ["recovery-review", rid, "--user-id", "dev"]
            human = runner.invoke(app, args)
            raw = runner.invoke(app, [*args, "--json"])
            hidden = runner.invoke(app, ["recovery-review", rid, "--user-id", "other"])
            assert human.exit_code == raw.exit_code == 0, human.output
            assert 'completion="budget_exhausted"' in human.output
            assert "+agent = 2" in json.loads(raw.output)["diffs"][0]["diff"]["text"]
            assert hidden.exit_code == 1 and "agent = 2" not in hidden.output
            assert gets.call_count == 3
            assert all(
                c.args[0].endswith(f"/runs/{rid}/recovery-review") for c in gets.call_args_list
            )
            post.assert_not_called()
        assert api.get(f"/runs/{rid}", params={"user_id": "dev"}).json() == saved
        assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archive
        assert len(api.get("/runs", params={"user_id": "dev"}).json()) == 1
    assert target.read_text() == "later user edits\n"
    assert restarted.calls == []
