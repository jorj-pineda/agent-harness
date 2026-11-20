"""Actual panel submission scripts over HTTP, simulated DOM/storage, scripted provider."""

import json
import shutil
import stat
import subprocess

import pytest

from providers.base import ToolCall
from tests.api.test_recovery_execution import RECOVERED, SETTINGS, observed_response, source
from tests.cli_process_support import ROOT, server
from tests.test_cli_process_sigint import wait_for


def panel(tmp_path, data, url, action):
    packet = tmp_path / "panel-packet.json"
    packet.write_text(
        json.dumps({"url": url, "candidate": data["candidate"], "delivery": data["delivery"]})
    )
    result = subprocess.run(
        [
            "node",
            str(ROOT / "tests/fixtures/panel_http_recovery.cjs"),
            str(ROOT / "ui"),
            str(packet),
            str(tmp_path / "panel-storage.json"),
            action,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node needed for panel HTTP contract")
@pytest.mark.parametrize("delivery", ["confirmed", "lost_response"])
def test_panel_posts_once_and_checks_saved_recovery_after_api_restart(tmp_path, delivery):
    data = source(tmp_path)  # In-process seed; submission/restart lookup use actual HTTP below.
    data["delivery"] = delivery
    data["candidate"] = {
        "source_run_id": data["rid"],
        "source_session_id": data["sid"],
        "draft": data["draft"],
        "input": {
            key: data["approval"][key]
            for key in ("user_id", "expected_record_sha256", "spans", "message")
        },
    }
    visible = (data["root"] / "test_a.py").read_bytes()
    settings = {
        **SETTINGS,
        "coding_toolset": "full",
        "memory_db_path": str(tmp_path / "memory.db"),
        "session_db_path": str(tmp_path / "sessions.db"),
    }
    replies = [
        observed_response(
            tool_calls=[
                ToolCall(
                    id="edit", name="write_file", arguments={"path": "a.py", "content": RECOVERED}
                )
            ]
        ).model_dump(mode="json"),
        observed_response(
            tool_calls=[
                ToolCall(
                    id="check",
                    name="run_command",
                    arguments={"argv": SETTINGS["project_check_argv"]},
                )
            ]
        ).model_dump(mode="json"),
        observed_response(content="Scripted HTTP recovery, no inference.").model_dump(mode="json"),
    ]
    stores = tmp_path / "stores"
    with server(tmp_path / "execute", stores, replies, overrides=settings) as (url, api):
        submitted = panel(tmp_path, data, url, "submit")
        assert submitted["calls"][0]["path"] == f"/runs/{data['rid']}/recover"
        retained = json.loads(submitted["retained"])
        assert json.loads(submitted["calls"][0]["body"]) == retained["body"]

        def finished():
            rows = api.get(
                "/runs", params={"user_id": "dev", "request_id": retained["body"]["request_id"]}
            ).json()
            assert len(rows) == 1
            return rows[0] if rows[0]["status"] != "running" else None

        record = wait_for(finished)
        assert record["response"]["completion_status"] == "completed", record
        assert record["response"]["verification_status"] == "passed"
        assert record["response"]["check_attempts"][0]["exit_code"] == 0
        assert record["request"]["session_id"] != data["sid"]
        assert api.get("/__preflight").json()["calls"] == 3
    with server(tmp_path / "restart", stores, [], overrides=settings) as (url, api):
        if delivery == "lost_response":
            replayed = panel(tmp_path, data, url, "retry")
            assert replayed["retained"] == submitted["retained"]
            assert replayed["calls"] == submitted["calls"]
            assert len(api.get("/runs", params={"user_id": "dev"}).json()) == 2
            assert api.get("/__preflight").json()["calls"] == 0
        reviewed = panel(tmp_path, data, url, "check")
        assert reviewed["retained"] == submitted["retained"]
        assert f"request_id={retained['body']['request_id']}" in reviewed["calls"][0]["path"]
        assert (
            json.loads(api.get(f"/runs/{record['run_id']}", params={"user_id": "dev"}).text)
            == record
        )
        assert api.get(f"/runs/{data['rid']}", params={"user_id": "dev"}).json() == data["record"]
        assert (
            api.get(f"/sessions/{data['sid']}", params={"user_id": "dev"}).json() == data["archive"]
        )
        assert api.get("/__preflight").json()["calls"] == 0
    assert (data["root"] / "a.py").read_text() == RECOVERED
    assert stat.S_IMODE((data["root"] / "a.py").stat().st_mode) == 0o640
    assert (data["root"] / "user.txt").read_text() == "pre-existing staged user work\n"
    assert (data["root"] / "test_a.py").read_bytes() == visible
    assert (data["root"] / ".git/index").read_bytes() == data["index"]
