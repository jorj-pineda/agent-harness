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
        json.dumps(
            {
                "url": url,
                "proposal": {k: data["approval"][k] for k in ("spans", "message")},
                "scenario": data.get("scenario"),
                "revert_id": data.get("revert_id"),
                "source": data["record"],
                "target": str(data["root"] / "a.py"),
            }
        )
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


def saved_source_settings(tmp_path):
    return {
        **SETTINGS,
        "coding_toolset": "full",
        "memory_db_path": str(tmp_path / "memory.db"),
        "session_db_path": str(tmp_path / "sessions.db"),
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node needed for panel HTTP contract")
def test_panel_loads_saved_source_evidence_without_executing(tmp_path):
    data = source(tmp_path)
    target = data["root"] / "a.py"
    original = target.read_bytes()
    with server(
        tmp_path / "review", tmp_path / "stores", [], overrides=saved_source_settings(tmp_path)
    ) as (url, api):
        panel(tmp_path, data, url, "review")
        assert api.get("/runs", params={"user_id": "dev"}).json() == [data["record"]]
        assert (
            api.get(f"/sessions/{data['sid']}", params={"user_id": "dev"}).json() == data["archive"]
        )
        assert api.get("/__preflight").json()["calls"] == 0
    assert target.read_bytes() == original
    assert (data["root"] / ".git/index").read_bytes() == data["index"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node needed for panel HTTP contract")
@pytest.mark.parametrize("scenario", ["confirmed", "lost_response", "cancelled", "edited_proposal"])
def test_panel_posts_once_and_checks_saved_recovery_after_api_restart(tmp_path, scenario):
    data = source(tmp_path)  # In-process seed; submission/restart lookup use actual HTTP below.
    data["scenario"] = scenario
    cancelled = scenario == "cancelled"
    visible = (data["root"] / "test_a.py").read_bytes()
    settings = saved_source_settings(tmp_path)
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
    if cancelled:
        replies = [replies[0], {"hang": True}]
    stores = tmp_path / "stores"
    with server(tmp_path / "execute", stores, replies, overrides=settings) as (url, api):
        submitted = panel(tmp_path, data, url, "submit")
        review, draft, *additional, dispatch = submitted["calls"]
        prefix = f"/runs/{data['rid']}"
        assert (review["method"], review["path"]) == (
            "GET",
            f"{prefix}/recovery-review?user_id=dev",
        )
        assert (draft["method"], draft["path"]) == ("POST", f"{prefix}/recovery-draft")
        assert json.loads(draft["body"]) == {
            k: data["approval"][k]
            for k in ("user_id", "expected_record_sha256", "spans", "message")
        }
        if scenario == "edited_proposal":
            (revised,) = additional
            assert (revised["method"], revised["path"]) == ("POST", f"{prefix}/recovery-draft")
            assert json.loads(revised["body"]) == {
                **json.loads(draft["body"]),
                "message": data["approval"]["message"] + " Keep the staged user file intact.",
            }
            assert json.loads(dispatch["body"])["message"] == json.loads(revised["body"])["message"]
        else:
            assert not additional
        assert dispatch["path"] == f"{prefix}/recover"
        retained = json.loads(submitted["retained"])
        assert json.loads(dispatch["body"]) == retained["body"]

        def finished():
            rows = api.get(
                "/runs", params={"user_id": "dev", "request_id": retained["body"]["request_id"]}
            ).json()
            assert len(rows) == 1
            return rows[0] if rows[0]["status"] != "running" else None

        if cancelled:
            wait_for(
                lambda: (
                    api.get("/__preflight").json()["calls"] == 2
                    and (data["root"] / "a.py").read_text() == RECOVERED
                )
            )
            running = api.get(
                "/runs", params={"user_id": "dev", "request_id": retained["body"]["request_id"]}
            ).json()[0]
            assert running["status"] == "running"
            cancellation = panel(tmp_path, data, url, "cancel")
            assert cancellation["retained"] == submitted["retained"]
            assert cancellation["calls"][0]["method"] == "GET"
            assert cancellation["calls"][1]["path"] == (
                f"/sessions/{running['request']['session_id']}/cancel"
            )
            assert json.loads(cancellation["calls"][1]["body"]) == {"user_id": "dev"}
        record = wait_for(finished)
        assert record["response"]["completion_status"] == (
            "cancelled" if cancelled else "completed"
        ), record
        assert record["response"]["verification_status"] == ("not_run" if cancelled else "passed")
        if cancelled:
            assert record["status"] == "cancelled"
            assert record["response"]["check_attempts"] == []
            assert record["response"]["token_usage"] == {
                "prompt_tokens": None,
                "completion_tokens": None,
            }
        else:
            assert record["response"]["check_attempts"][0]["exit_code"] == 0
        assert record["request"]["session_id"] != data["sid"]
        assert api.get("/__preflight").json()["calls"] == (2 if cancelled else 3)
    expected_user = "pre-existing staged user work\n"
    if cancelled:
        expected_user += "later unrelated user edit\n"
        (data["root"] / "user.txt").write_text(expected_user)
    with server(tmp_path / "restart", stores, [], overrides=settings) as (url, api):
        if scenario == "lost_response":
            replayed = panel(tmp_path, data, url, "retry")
            assert replayed["retained"] == submitted["retained"]
            assert replayed["calls"] == [dispatch]
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
    assert (data["root"] / "user.txt").read_text() == expected_user
    assert (data["root"] / "test_a.py").read_bytes() == visible
    assert (data["root"] / ".git/index").read_bytes() == data["index"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node needed for panel HTTP contract")
@pytest.mark.parametrize(
    ("scenario", "retry"),
    [
        ("applied", False),
        ("stale_reversal", False),
        ("lost_reversal_response", False),
        pytest.param("lost_reversal_response", True, id="lost_response_explicit_retry"),
    ],
)
def test_panel_reverses_one_saved_file_over_http(tmp_path, scenario, retry):
    data = source(tmp_path)
    data["scenario"] = scenario
    stale = scenario == "stale_reversal"
    root = data["root"]
    user = root / "user.txt"
    user.write_text("later unrelated user edit\n")
    with server(
        tmp_path / "reverse", tmp_path / "stores", [], overrides=saved_source_settings(tmp_path)
    ) as (url, api):
        result = panel(tmp_path, data, url, "reverse")
        retained = json.loads(result["retained"])
        assert retained["run_id"] == data["rid"]
        assert retained["path"] == "a.py"
        saved = api.get(
            "/file-reverts", params={"user_id": "dev", "request_id": retained["request_id"]}
        ).json()
        assert saved["request"] == retained
        assert saved["status"] == ("rejected" if stale else "applied")
        assert api.get("/runs", params={"user_id": "dev"}).json() == [data["record"]]
        assert (
            api.get(f"/sessions/{data['sid']}", params={"user_id": "dev"}).json() == data["archive"]
        )
        assert api.get("/__preflight").json()["calls"] == 0
    target = root / "a.py"
    assert target.read_text() == (
        "later user edit to the reversal target\n" if stale else "user = 1\n"
    )
    assert stat.S_IMODE(target.stat().st_mode) == (0o600 if stale else 0o640)
    assert user.read_text() == "later unrelated user edit\n"
    assert (root / ".git/index").read_bytes() == data["index"]

    target.write_text("user edit after reversal settled\n")
    target.chmod(0o604)
    target_bytes = target.read_bytes()
    user_bytes = user.read_bytes()
    with server(
        tmp_path / "reverse-restart",
        tmp_path / "stores",
        [],
        overrides=saved_source_settings(tmp_path),
    ) as (url, api):
        if retry:
            data["revert_id"] = saved["revert_id"]
            replayed = panel(tmp_path, data, url, "reverse-retry")
            assert replayed["retained"] == result["retained"]
            assert replayed["calls"] == [result["calls"][1]]
            assert target.read_bytes() == target_bytes
            assert stat.S_IMODE(target.stat().st_mode) == 0o604
        checked = panel(tmp_path, data, url, "reverse-check")
        assert checked["retained"] == result["retained"]
        assert (
            api.get(
                "/file-reverts", params={"user_id": "dev", "request_id": retained["request_id"]}
            ).json()
            == saved
        )
        assert api.get("/runs", params={"user_id": "dev"}).json() == [data["record"]]
        assert (
            api.get(f"/sessions/{data['sid']}", params={"user_id": "dev"}).json() == data["archive"]
        )
        assert api.get("/__preflight").json()["calls"] == 0
    assert target.read_bytes() == target_bytes
    assert stat.S_IMODE(target.stat().st_mode) == 0o604
    assert user.read_bytes() == user_bytes
    assert (root / ".git/index").read_bytes() == data["index"]
