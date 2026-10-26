import hashlib
import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli._client import AgentClient
from cli.main import app
from tests.api.conftest import ScriptedProvider
from tests.api.test_session_persistence import make_app

runner = CliRunner()
HASH = "a" * 64


def data():
    content = "one\r\n🙂two\x1b[2J\n"
    return {
        "schema_version": 1,
        "session_id": "s1",
        "checkpoint_id": "c1",
        "checkpoint_kind": "manual",
        "checkpoint_record_sha256": "b" * 64,
        "path": "a.txt",
        "file_sha256": HASH,
        "file_bytes": 100,
        "mode": 0o755,
        "total_lines": 3,
        "start_line": 1,
        "end_line": 2,
        "selected_bytes": len(content.encode()),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "content": content,
        "has_more": True,
        "read_only": True,
        "scope_note": "saved untrusted source\x1b[2J",
    }


def args(sid="s1", cid="c1", expected=HASH):
    return [
        "checkpoint-span",
        sid,
        cid,
        "--path",
        "a.txt",
        "--expected-sha256",
        expected,
        "--start-line",
        "1",
        "--end-line",
        "2",
        "--user-id",
        "dev",
    ]


def response(value, status=200):
    return httpx.Response(status, json=value, request=httpx.Request("GET", "http://fixture"))


def test_human_and_json_preserve_exact_source_and_safe_metadata_without_other_requests():
    value = data()
    with patch("httpx.get", return_value=response(value)) as get, patch("httpx.post") as post:
        human = runner.invoke(app, args())
        raw = runner.invoke(app, [*args(), "--json"])
    assert human.exit_code == raw.exit_code == 0, human.output
    assert "Historical source only" in human.output
    assert "lines=1..2 total_lines=3 has_more=true" in human.output
    assert "saved_mode=0755" in human.output
    assert "\\u001b" in human.output and "\x1b" not in human.output
    assert "\\r\\n" in human.output
    assert json.loads(raw.output) == value
    assert get.call_count == 2
    get.assert_called_with(
        "http://localhost:8000/sessions/s1/checkpoints/c1/span",
        params={
            "user_id": "dev",
            "path": "a.txt",
            "expected_sha256": HASH,
            "start_line": 1,
            "end_line": 2,
        },
        timeout=30,
    )
    post.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("session_id", "other"),
        ("checkpoint_id", "other"),
        ("path", "other"),
        ("file_sha256", "0" * 64),
        ("start_line", True),
        ("end_line", 3),
        ("selected_bytes", 0),
        ("selected_bytes", False),
        ("file_bytes", 1),
        ("total_lines", 1),
        ("has_more", False),
        ("mode", True),
        ("content_sha256", "0" * 64),
        ("checkpoint_record_sha256", "bad"),
        ("read_only", False),
        ("checkpoint_kind", "verified"),
        ("scope_note", None),
        ("content", None),
    ],
)
@pytest.mark.parametrize("raw", [False, True])
def test_invalid_identity_counts_or_content_fail_before_either_output(field, value, raw):
    record = data()
    record[field] = value
    with patch("httpx.get", return_value=response(record)) as get, patch("httpx.post") as post:
        result = runner.invoke(app, [*args(), *(["--json"] if raw else [])])
    assert result.exit_code == 1
    assert "saved_checkpoint_span" not in result.output and "one" not in result.output
    assert "No automatic retry" in result.output
    assert get.call_count == 1
    post.assert_not_called()


def test_content_line_count_is_checked_even_with_matching_bytes_and_hash():
    record = data()
    record["content"] = "one line"
    record["selected_bytes"] = 8
    record["content_sha256"] = hashlib.sha256(b"one line").hexdigest()
    with patch("httpx.get", return_value=response(record)):
        result = runner.invoke(app, [*args(), "--json"])
    assert result.exit_code == 1 and "one line" not in result.output


@pytest.mark.parametrize("status", [403, 404, 409, 413, 415, 422, 503])
def test_server_refusals_never_print_private_bodies_or_retry(status):
    with (
        patch("httpx.get", return_value=response({"detail": "private-marker"}, status)) as get,
        patch("httpx.post") as post,
    ):
        result = runner.invoke(app, args())
    assert result.exit_code == 1 and "private-marker" not in result.output
    assert get.call_count == 1
    post.assert_not_called()


@pytest.mark.parametrize(
    "failure", [httpx.ReadTimeout("private-marker"), ValueError("private-marker")]
)
def test_transport_errors_remain_private_and_never_fall_back(failure):
    with patch("httpx.get", side_effect=failure) as get, patch("httpx.post") as post:
        result = runner.invoke(app, args())
    assert result.exit_code == 1 and "private-marker" not in result.output
    assert get.call_count == 1
    post.assert_not_called()


@pytest.mark.parametrize(
    "start,end,expected",
    [(0, 2, HASH), (3, 2, HASH), (1, 201, HASH), (True, 2, HASH), (1, 2, "bad")],
)
def test_invalid_requests_do_not_dispatch(start, end, expected):
    with patch("httpx.get") as get, pytest.raises(ValueError):
        AgentClient().checkpoint_span(
            "dev",
            "s1",
            "c1",
            path="a.txt",
            expected_sha256=expected,
            start_line=start,
            end_line=end,
        )
    get.assert_not_called()


@pytest.mark.parametrize("claim", ["nonfinal_terminator", "whole_file"])
def test_range_claims_cannot_contradict_lf_boundaries_or_full_file_identity(claim):
    record = data()
    if claim == "nonfinal_terminator":
        record["content"] = "first\nsecond"
        record["selected_bytes"] = len(record["content"].encode())
        record["content_sha256"] = hashlib.sha256(record["content"].encode()).hexdigest()
    else:
        record.update(total_lines=2, has_more=False, file_bytes=record["selected_bytes"])
    with patch("httpx.get", return_value=response(record)) as get:
        result = runner.invoke(app, [*args(), "--json"])
    assert result.exit_code == 1 and "saved untrusted source" not in result.output
    assert get.call_count == 1


def test_oversized_wire_body_refuses_before_printing():
    with patch("httpx.get", return_value=response({**data(), "extra": "x" * 32768})) as get:
        result = runner.invoke(app, [*args(), "--json"])
    assert result.exit_code == 1 and "one" not in result.output and get.call_count == 1


def test_real_cli_after_restart_reads_saved_bytes_not_later_user_edits(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "a.txt"
    source = "saved\r\n🙂second\n"
    target.write_bytes(source.encode())
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider)) as api:
        sid = api.post("/sessions", json={"user_id": "dev", "workspace_root": str(root)}).json()[
            "session_id"
        ]
        record = api.post(f"/sessions/{sid}/checkpoints", json={"user_id": "dev"}).json()
        archive = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
    target.write_text("later user edit\n")
    root.rename(tmp_path / "moved")
    with TestClient(make_app(tmp_path, provider)) as api:

        def get(url, **kwargs):
            assert kwargs["timeout"] == 30
            return api.get(url.removeprefix("http://localhost:8000"), params=kwargs["params"])

        with patch("httpx.get", side_effect=get) as gets, patch("httpx.post") as post:
            requested = args(sid, record["checkpoint_id"], record["files"]["a.txt"]["sha256"])
            human = runner.invoke(app, requested)
            raw = runner.invoke(app, [*requested, "--json"])
            hidden = runner.invoke(app, [*requested, "--user-id", "other"])
            assert human.exit_code == raw.exit_code == 0, human.output
            assert json.loads(raw.output)["content"] == source
            assert hidden.exit_code == 1 and "saved\\r" not in hidden.output
            assert gets.call_count == 3
            assert all(
                c.args[0].endswith(f"/checkpoints/{record['checkpoint_id']}/span")
                for c in gets.call_args_list
            )
            post.assert_not_called()
        assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archive
        assert (
            api.get(
                f"/sessions/{sid}/checkpoints/{record['checkpoint_id']}", params={"user_id": "dev"}
            ).json()
            == record
        )
        assert api.get("/runs", params={"user_id": "dev"}).json() == []
    assert (tmp_path / "moved/a.txt").read_text() == "later user edit\n"
    assert provider.calls == []
