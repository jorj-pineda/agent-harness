"""Real local HTTP/CLI/files/SQLite, scripted inference; never a quality cohort."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from typer.testing import CliRunner

from cli.main import app
from evals.projects.duodoro.recovery_lifecycle import OwnedAPIs
from evals.projects.duodoro.recovery_staging import OwnedWorkspace, clone_closed_stores, seal_tree
from evals.real_run import Task
from harness.execution_profile import tool_specs_digest
from harness.recovery_identity import recovery_messages_sha256
from providers.base import ChatMessage, ToolCall, ToolSpec
from tests.api.conftest import make_response

ROOT = Path(__file__).resolve().parents[2]
PLAN = json.loads((ROOT / "evals/projects/duodoro/projected-recovery-plan.json").read_text())
MESSAGE = PLAN["followup_message"]
CHECK = PLAN["settings"]["project_check_argv"]
OWNER = "preflight"
runner = CliRunner()


def reply(content="", *, tools=(), known=True):
    return make_response(
        content=content,
        tool_calls=list(tools),
        prompt_tokens=100 if known else None,
        completion_tokens=20 if known else None,
    ).model_dump(mode="json")


def write(value):
    return ToolCall(
        id="edit", name="write_file", arguments={"path": "a.py", "content": f"value = {value}\n"}
    )


def settings(stores):
    return {
        **PLAN["settings"],
        **PLAN["phase_limits"],
        "default_provider": "ollama",
        "ollama_model": "gemma4:12b",
        "memory_db_path": str(stores / "facts.sqlite"),
        "session_db_path": str(stores / "sessions.sqlite"),
    }


def wait_for(check, *, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.025)
    raise AssertionError("Bounded local preflight condition did not settle")


@contextmanager
def server(directory, stores, responses, resources, *, overrides=None):
    directory.mkdir(mode=0o700)
    configuration = directory / "configuration.json"
    configuration.write_text(
        json.dumps({"settings": {**settings(stores), **(overrides or {})}, "responses": responses})
    )
    configuration.chmod(0o600)
    trace = directory / "trace.json"
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    environment = {
        k: os.environ[k] for k in ("PATH", "VIRTUAL_ENV", "LANG", "TMPDIR") if k in os.environ
    }
    with (directory / "server.log").open("wb") as log:
        os.chmod(directory / "server.log", 0o600)
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "tests/fixtures/paired_recovery_server.py"),
                str(configuration),
                str(trace),
                str(port),
            ],
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        gate = resources
        try:
            resources.register(process, port)
            with httpx.Client(base_url=url, timeout=10) as client:

                def ready():
                    assert process.poll() is None, (directory / "server.log").read_text()
                    try:
                        response = client.get("/__preflight")
                    except httpx.ConnectError:
                        return False
                    return response.status_code == 200 and response.json()["pid"] == process.pid

                wait_for(ready)
                yield url, client, gate
        finally:
            if process.poll() is None:
                process.terminate()  # This exact owned child only; normal lifespan shutdown.
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                raise AssertionError("Owned preflight API failed graceful shutdown") from None
        gate()
    saved = json.loads(trace.read_text())
    assert saved["closing"] and saved["stores_closed"] and saved["unfinished_tasks"] == 0
    assert saved["inference_ran"] is False


def cli(url, arguments, *, input=None, ok=True):
    result = runner.invoke(app, [*arguments, "--base-url", url, "--user-id", OWNER], input=input)
    if ok:
        assert result.exit_code == 0, result.output
    return result


def get(client, path, **params):
    response = client.get(path, params={"user_id": OWNER, **params})
    assert response.status_code == 200, response.text
    return response.json()


def terminal(client, rid):
    def lookup():
        record = get(client, f"/runs/{rid}")
        return record if record["status"] != "running" else None

    return wait_for(lookup)


def hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


@pytest.fixture
def resources():
    return OwnedAPIs()


@pytest.fixture
def prepared(tmp_path, request, resources):
    fixture, acceptance, reference = (
        tmp_path / name for name in ("fixture", "acceptance", "reference")
    )
    for path in (fixture, acceptance, reference):
        path.mkdir()
    (fixture / "a.py").write_text("value = -1\n")
    (fixture / "a.py").chmod(0o640)
    (fixture / "notes.txt").write_text("preserve setup work\n")
    (fixture / "test_visible.py").write_text(
        "from a import value\ndef test_value():\n    assert value == 1\n"
    )
    if getattr(request, "param", None) == "slow":
        (fixture / "test_visible.py").write_text(
            "import os, time\nfrom pathlib import Path\n"
            f"Path({str(tmp_path / 'owned-pytest-pid')!r}).write_text(str(os.getpid()))\n"
            "time.sleep(30)\ndef test_value():\n    assert True\n"
        )
    # Real tool history can exceed the full-history estimate while projection fits.
    (fixture / "large.txt").write_text("z" * 55_000 + "\n")
    (acceptance / "test_acceptance.py").write_text(
        "from pathlib import Path\ndef test_value():\n    assert Path('a.py').read_text() == 'value = 1\\n'\n    assert Path('notes.txt').read_text() == 'preserve setup work\\n'\n"
    )
    (reference / "a.py").write_text("value = 1\n")
    task = Task(
        "portable-paired-preflight",
        "Inspect a.py, set value to 1, preserve setup work and run the configured check.",
        fixture,
        acceptance,
        reference,
    )
    protected = tuple(
        seal_tree(p, tmp_path / ("protected-" + p.name)) for p in (fixture, acceptance, reference)
    )
    workspace = OwnedWorkspace.create(tmp_path / "owned")
    workspace.stage(protected[0], last_scored=None, protected=protected, settled=resources)
    return workspace, task, protected


def source(tmp_path, prepared, resources, *, crowded=False):
    workspace, task, protected = prepared
    stores = tmp_path / "source-stores"
    stores.mkdir()
    tools = [
        write(0),
        ToolCall(
            id="fact",
            name="remember_fact",
            arguments={"fact": "Preserve setup work in this workspace."},
        ),
    ]
    if crowded:
        tools.append(ToolCall(id="large", name="read_file", arguments={"path": "large.txt"}))
    with server(
        tmp_path / "source-api", stores, [reply(tools=tools), reply(), reply()], resources
    ) as (
        url,
        client,
        gate,
    ):
        result = cli(
            url,
            [
                "chat",
                "--workspace",
                str(workspace.root),
                "--checkpoints",
                "--response-timeout",
                "360",
            ],
            input=task.request + "\n/quit\n",
        )
        match = re.search(r"session=([0-9a-f]{32})", result.output)
        assert match is not None, result.output
        sid = match.group(1)
        records = get(client, "/runs", session_id=sid)
        assert len(records) == 1
        record = records[0]
        assert record["response"]["completion_status"] in ("incomplete", "budget_exhausted")
        assert record["before_checkpoint_id"] and record["after_checkpoint_id"]
        archive = get(client, f"/sessions/{sid}")
    preserved = asyncio.run(
        workspace.score(task, tmp_path / "source-artifact", protected=protected, settled=gate)
    )
    parent = tmp_path / "parent-stores"
    expected = clone_closed_stores(stores, parent, expected=hashes(stores), settled=gate)
    return {
        "sid": sid,
        "rid": record["run_id"],
        "record": record,
        "archive": archive,
        "artifact": preserved,
        "gate": gate,
        "parent": parent,
        "hashes": expected,
        "stores": stores,
    }


def projected_args(url, client, data, directory):
    review = json.loads(cli(url, ["recovery-review", data["rid"], "--json"]).output)
    checkpoint = get(
        client, f"/sessions/{data['sid']}/checkpoints/{data['record']['after_checkpoint_id']}"
    )
    selection = {
        "expected_record_sha256": review["source_record_sha256"],
        "spans": [
            {
                "path": "a.py",
                "expected_sha256": checkpoint["files"]["a.py"]["sha256"],
                "start_line": 1,
                "end_line": 1,
            }
        ],
    }
    selection_file = directory / "selection.json"
    selection_file.write_text(json.dumps(selection))
    selection_file.chmod(0o600)
    draft = json.loads(
        cli(
            url,
            [
                "recovery-draft",
                data["rid"],
                "--selection-file",
                str(selection_file),
                "--message",
                MESSAGE,
                "--json",
            ],
        ).output
    )
    assert draft["configuration"]["status"] == "matching"
    assert draft["context"]["status"] == "estimated_fit"
    approval = {
        **selection,
        "message": MESSAGE,
        "expected_archive_sha256": draft["source_archive_sha256"],
        "expected_messages_sha256": draft["messages_sha256"],
        "expected_tool_specs_sha256": draft["tool_specs_sha256"],
        "expected_request_output_allowance": draft["context"]["request_output_allowance"],
    }
    path = directory / "approval.json"
    path.write_text(json.dumps(approval))
    path.chmod(0o600)
    return [
        "recover-run",
        data["rid"],
        "--approval-file",
        str(path),
        "--request-id",
        "projected-once",
        "--acknowledge-uncaptured-state",
        "--acknowledge-omitted-context",
        "--acknowledge-new-turn-budget",
        "--json",
    ], draft


def followup():
    return [
        reply(tools=[write(1)]),
        reply(tools=[ToolCall(id="check", name="run_command", arguments={"argv": CHECK})]),
        reply("Changed and checked; scripted preflight only."),
    ]


@pytest.mark.parametrize(
    "order,crowded", [(("C", "P"), False), (("P", "C"), False), (("C", "P"), True)]
)
def test_paired_actual_cli_http_staging_same_source_both_orders_and_context_refusal(
    tmp_path, prepared, order, crowded, resources
):
    workspace, task, protected = prepared
    data = source(tmp_path, prepared, resources, crowded=crowded)
    previous, previous_gate = data["artifact"], data["gate"]
    outcomes = {}
    for arm in order:
        workspace.stage(
            data["artifact"].tree, last_scored=previous, protected=protected, settled=previous_gate
        )
        stores = tmp_path / (arm + "-stores")
        clone_closed_stores(data["parent"], stores, expected=data["hashes"], settled=previous_gate)
        directory = tmp_path / (arm + "-api")
        draft = None
        with server(directory, stores, followup(), resources) as (url, client, gate):
            with pytest.raises(ValueError, match="active"):
                gate()
            with pytest.raises(ValueError, match="active"):
                workspace.stage(
                    data["artifact"].tree,
                    last_scored=previous,
                    protected=protected,
                    settled=previous_gate,
                )
            if arm == "C":
                arguments = [
                    "continue-session",
                    data["sid"],
                    MESSAGE,
                    "--expected-turn-id",
                    data["archive"]["session"]["turns"][-1]["turn_id"],
                    "--request-id",
                    "control-once",
                    "--acknowledge-uncaptured-state",
                    "--json",
                ]
            else:
                arguments, draft = projected_args(url, client, data, directory)
            submitted = cli(url, arguments, ok=False)
            if crowded and arm == "C":
                assert submitted.exit_code == 0, submitted.output
                blocked = terminal(client, json.loads(submitted.output)["run_id"])
                assert blocked["response"]["completion_status"] == "blocked"
                assert "context fit" in blocked["response"]["completion_reason"]
                assert len(get(client, "/runs")) == 2
                outcomes[arm] = "blocked"
            else:
                assert submitted.exit_code == 0, submitted.output
                rid = json.loads(submitted.output)["run_id"]
                run = terminal(client, rid)
                assert run["response"]["completion_status"] == "completed", run
                assert run["response"]["verification_status"] == "passed"
                assert run["request"]["message"] == MESSAGE
                assert run["request"]["continuation_source"]["session_id"] == data["sid"]
                outcomes[arm] = "completed"
            assert get(client, f"/runs/{data['rid']}") == data["record"]
            assert get(client, f"/sessions/{data['sid']}") == data["archive"]
        trace = json.loads((directory / "trace.json").read_text())
        if outcomes[arm] == "blocked":
            assert trace["calls"] == []
        else:
            assert len(trace["calls"]) == 3
            first = trace["calls"][0]
            assert first["messages"][-1]["content"] == MESSAGE
            assert "Preserve setup work in this workspace." in first["messages"][0]["content"]
            if draft is not None:
                assert (
                    recovery_messages_sha256(
                        [ChatMessage.model_validate(m) for m in first["messages"]]
                    )
                    == draft["messages_sha256"]
                )
                assert first["max_tokens"] == draft["context"]["request_output_allowance"]
                assert (
                    tool_specs_digest([ToolSpec.model_validate(t) for t in first["tools"]])
                    == draft["tool_specs_sha256"]
                )
                assert not any(m["role"] == "tool" for m in first["messages"])
            else:
                assert any(m["role"] == "tool" for m in first["messages"])
        previous = asyncio.run(
            workspace.score(task, tmp_path / (arm + "-artifact"), protected=protected, settled=gate)
        )
        previous_gate = gate
        assert hashes(data["parent"]) == data["hashes"]
        assert hashes(data["stores"]) == data["hashes"]
    assert outcomes == {"C": "blocked" if crowded else "completed", "P": "completed"}
    assert workspace.root.stat().st_ino == workspace.identity[1]


@pytest.mark.parametrize(
    "failure,prepared",
    [
        (name, "normal")
        for name in ("failed_check", "unknown_usage", "provider_failure", "cancel", "lost_response")
    ]
    + [("cancel_tools", "slow")],
    indirect=["prepared"],
)
def test_projected_failures_and_restart_lookup_never_resubmit(
    tmp_path, prepared, failure, resources
):
    workspace, task, protected = prepared
    data = source(tmp_path, prepared, resources)
    stores = tmp_path / "arm-stores"
    clone_closed_stores(data["parent"], stores, expected=data["hashes"], settled=data["gate"])
    responses = followup()
    if failure == "failed_check":
        responses[0] = reply(tools=[write(0)])
        responses.extend([reply(), reply()])
    elif failure == "unknown_usage":
        responses = [reply(tools=[write(1)], known=False)]
    elif failure == "provider_failure":
        responses = [{"failure": True}]
    elif failure == "cancel":
        responses = [reply(tools=[write(1)]), {"hang": True}]
    directory = tmp_path / "arm-api"
    with server(directory, stores, responses, resources) as (url, client, _):
        arguments, _ = projected_args(url, client, data, directory)
        real_post = httpx.post
        if failure == "lost_response":

            def lost(url, **kwargs):
                result = real_post(url, **kwargs)
                assert result.status_code == 202
                raise httpx.ReadTimeout("Scripted lost acknowledgment")

            with patch("httpx.post", side_effect=lost) as posted:
                result = cli(url, arguments, ok=False)
            assert result.exit_code == 1 and posted.call_count == 1
            assert "projected-once" in result.output
            # Original-key GET only: never retry or replace this submission.
            assert (
                "request_id=projected-once"
                in cli(url, ["runs", "--request-id", "projected-once"]).output
            )
            submitted = get(client, "/runs", request_id="projected-once")[0]
        else:
            submitted = json.loads(cli(url, arguments).output)
        rid = submitted["run_id"]
        if failure in ("cancel", "cancel_tools"):
            wait_for(lambda: (workspace.root / "a.py").read_text() == "value = 1\n")
            if failure == "cancel_tools":
                wait_for(lambda: (tmp_path / "owned-pytest-pid").exists())
            else:
                wait_for(lambda: client.get("/__preflight").json()["calls"] == 2)
            cli(url, ["cancel", submitted["request"]["session_id"]])
        final = terminal(client, rid)
        if failure == "cancel_tools":
            # Cancel acknowledges the request; the terminal turn owns cleanup.
            pid = int((tmp_path / "owned-pytest-pid").read_text())
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        assert get(client, f"/runs/{data['rid']}") == data["record"]
        assert get(client, f"/sessions/{data['sid']}") == data["archive"]
        if failure == "provider_failure":
            assert final["status"] == "failed" and final["response"] is None
        else:
            response = final["response"]
            assert (
                response["completion_status"]
                == {
                    "failed_check": "incomplete",
                    "unknown_usage": "incomplete",
                    "cancel": "cancelled",
                    "cancel_tools": "cancelled",
                    "lost_response": "completed",
                }[failure]
            )
            if failure == "failed_check":
                assert response["verification_status"] == "failed"
            if failure in ("unknown_usage", "cancel"):
                assert response["token_usage"]["completion_tokens"] is None
            if failure == "unknown_usage":
                assert (workspace.root / "a.py").read_text() == "value = 0\n"
    arm_artifact = asyncio.run(
        workspace.score(task, tmp_path / "arm-artifact", protected=protected, settled=resources)
    )
    grades = json.loads((arm_artifact.tree.directory / "grades.json").read_text())
    expected_grade = 0 if failure in ("lost_response", "cancel", "cancel_tools") else 1
    assert [grade["exit_code"] for grade in grades] == [expected_grade, expected_grade]
    (workspace.root / "notes.txt").write_text("later user work; retain")
    with server(tmp_path / "restart-api", stores, [], resources) as (url, client, _):
        with patch("httpx.post") as posted:
            assert json.loads(cli(url, ["run", rid, "--json"]).output) == final
            cli(url, ["runs", "--request-id", "projected-once"])
        posted.assert_not_called()
        assert get(client, f"/runs/{data['rid']}") == data["record"]
    assert json.loads((tmp_path / "restart-api/trace.json").read_text())["calls"] == []
    assert (workspace.root / "notes.txt").read_text() == "later user work; retain"
    assert hashes(data["parent"]) == data["hashes"]
    with pytest.raises(ValueError, match="Last-scored"):
        workspace.stage(
            data["artifact"].tree, last_scored=arm_artifact, protected=protected, settled=resources
        )
    assert (workspace.root / "notes.txt").read_text() == "later user work; retain"
