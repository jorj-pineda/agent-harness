"""One explicit historical preview POST, no continuation or implicit metadata fetch."""

import copy
import hashlib
import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli.main import app
from harness.recovery_context import build_recovery_context
from tests.api.conftest import ScriptedProvider
from tests.api.test_recovery_context import seed
from tests.api.test_session_persistence import make_app
from tests.test_recovery_context import fixture, selection

runner = CliRunner()


def prepared(tmp_path, missing_final=False):
    _, store, record, _, span = fixture(tmp_path, b"first\r\nsecond\x1b[2J\nlast")
    record.request.message = "Fix\x1b[2J"
    if missing_final:
        record.response = None
        record.status = "interrupted"
    request = selection(
        record,
        [] if missing_final else [span, span.model_copy(update={"start_line": 2, "end_line": 2})],
    )
    data = build_recovery_context(store, record, request).model_dump(mode="json")
    store.close()
    path = tmp_path / "selection.json"
    path.write_text(request.model_dump_json())
    return record.run_id, request.model_dump(mode="json"), path, data


def args(rid, path):
    return ["recovery-context", rid, "--selection-file", str(path), "--user-id", "dev"]


def response(data, status=200):
    return httpx.Response(status, json=data, request=httpx.Request("POST", "http://fixture"))


def rehash(data):
    def compact(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    evidence = compact(data["review"]) + "\n" + "\n".join(compact(s) for s in data["spans"])
    data["evidence_sha256"] = hashlib.sha256(evidence.encode()).hexdigest()


def test_human_and_json_preserve_exact_evidence_and_make_only_one_preview_request(tmp_path):
    rid, request, path, data = prepared(tmp_path)
    before = path.read_bytes()
    with patch("httpx.post", return_value=response(data)) as post, patch("httpx.get") as get:
        human = runner.invoke(app, args(rid, path))
        raw = runner.invoke(app, [*args(rid, path), "--json"])
    assert human.exit_code == raw.exit_code == 0, human.output
    assert 'completion="budget_exhausted" checks="failed"' in human.output
    assert "execution_supported=false" in human.output and "spans=2" in human.output
    assert "\\u001b" in human.output and "\x1b" not in human.output
    assert "\\r\\n" in human.output
    assert '"diff":' in human.output and '"status": "failed"' in human.output
    assert json.loads(raw.output) == data
    assert post.call_count == 2
    post.assert_called_with(
        f"http://localhost:8000/runs/{rid}/recovery-context",
        json={"user_id": "dev", **request},
        timeout=30,
    )
    get.assert_not_called()
    assert path.read_bytes() == before


def test_zero_span_missing_final_keeps_unknown_counts(tmp_path):
    rid, _, path, data = prepared(tmp_path, missing_final=True)
    with patch("httpx.post", return_value=response(data)):
        result = runner.invoke(app, args(rid, path))
    assert result.exit_code == 0, result.output
    assert "completion=unknown checks=unknown" in result.output
    assert (
        "action_count=unknown" in result.output
        and "selected_source_bytes=0 spans=0" in result.output
    )


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "readonly",
        "execute",
        "countbool",
        "sourcecount",
        "sourcesize",
        "envelopehash",
        "run",
        "runhash",
        "active",
        "unknowncounts",
        "selectedcount",
        "spankind",
        "mixedcheckpoint",
        "mixedsession",
        "mixedrecordhash",
        "spanhash",
        "contenthash",
        "linecount",
        "range",
        "mode",
        "nonfinal",
        "omissions",
        "scope",
        "wire",
    ],
)
@pytest.mark.parametrize("raw", [False, True])
def test_invalid_response_refuses_before_printing_any_evidence(tmp_path, change, raw):
    rid, _, path, data = prepared(tmp_path)
    if change == "version":
        data["schema_version"] = True
    elif change == "readonly":
        data["read_only"] = False
    elif change == "execute":
        data["execution_supported"] = 0  # False must not accept numeric zero.
    elif change == "countbool":
        data["selected_source_bytes"] = True
    elif change == "sourcecount":
        data["selected_source_bytes"] += 1
    elif change == "sourcesize":
        data["selected_source_bytes"] = 16385
    elif change == "envelopehash":
        data["evidence_sha256"] = "0" * 64
    elif change == "run":
        data["review"]["run_id"] = "foreign"
    elif change == "runhash":
        data["review"]["source_record_sha256"] = "0" * 64
    elif change == "active":
        data["review"]["run_status"] = "running"
    elif change == "unknowncounts":
        data["review"]["action_count"] = None
    elif change == "selectedcount":
        data["spans"].pop()
    elif change == "spankind":
        data["spans"][0]["checkpoint_kind"] = "manual"
    elif change == "mixedcheckpoint":
        data["spans"][1]["checkpoint_id"] = "foreign"
    elif change == "mixedsession":
        data["spans"][1]["session_id"] = "foreign"
    elif change == "mixedrecordhash":
        data["spans"][1]["checkpoint_record_sha256"] = "f" * 64
    elif change == "spanhash":
        data["spans"][0]["file_sha256"] = "0" * 64
    elif change == "contenthash":
        data["spans"][0]["content_sha256"] = "0" * 64
    elif change == "linecount":
        data["spans"][0]["content"] = "two\nlines\n"
        data["spans"][0]["selected_bytes"] = 10
        data["spans"][0]["content_sha256"] = hashlib.sha256(b"two\nlines\n").hexdigest()
    elif change == "range":
        data["spans"][0]["start_line"] = True
    elif change == "mode":
        data["spans"][0]["mode"] = False
    elif change == "nonfinal":
        span = data["spans"][0]
        span["content"] = "first"
        span["selected_bytes"] = 5
        span["content_sha256"] = hashlib.sha256(b"first").hexdigest()
    elif change == "omissions":
        data["review"]["omitted_diffs"] = 1
    elif change == "scope":
        data["scope_note"] = None
    elif change == "wire":
        data["extra"] = "private-marker" * 65536
    if change != "envelopehash":
        rehash(data)  # Consistent aggregate hash cannot bypass structural/source checks.
    with patch("httpx.post", return_value=response(data)) as post, patch("httpx.get") as get:
        result = runner.invoke(app, [*args(rid, path), *(["--json"] if raw else [])])
    assert result.exit_code == 1
    assert "No automatic retry" in result.output
    assert "recovery_review run_id=" not in result.output and "first" not in result.output
    assert "private-marker" not in result.output
    assert post.call_count == 1
    get.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "spans",
        "extra",
        "range",
        "count",
        "overlap",
        "spanextra",
        "path",
        "spanhash",
    ],
)
def test_invalid_selection_refuses_before_http(tmp_path, change):
    rid, request, path, _ = prepared(tmp_path)
    if change == "hash":
        request["expected_record_sha256"] = "bad"
    elif change == "spans":
        request["spans"] = None
    elif change == "extra":
        request["execute"] = True
    elif change == "range":
        request["spans"][0]["start_line"] = True
    elif change == "count":
        request["spans"] *= 3
    elif change == "overlap":
        request["spans"][1] = copy.deepcopy(request["spans"][0])
    elif change == "spanextra":
        request["spans"][0]["extra"] = True
    elif change == "path":
        request["spans"][0]["path"] = ""
    else:
        request["spans"][0]["expected_sha256"] = "bad"
    path.write_text(json.dumps(request))
    with patch("httpx.post") as post, patch("httpx.get") as get:
        result = runner.invoke(app, args(rid, path))
    assert result.exit_code == 1
    post.assert_not_called()
    get.assert_not_called()


@pytest.mark.parametrize(
    "content",
    [
        b"x" * 8193,
        b"\xff",
        b"[]",
        b'{"spans":[],"spans":[]}',
        b"private-marker",
    ],
)
def test_invalid_or_oversized_selection_files_do_not_dispatch(tmp_path, content):
    path = tmp_path / "selection.json"
    path.write_bytes(content)
    with patch("httpx.post") as post:
        result = runner.invoke(app, args("run", path))
    assert result.exit_code == 1 and "private-marker" not in result.output
    assert path.read_bytes() == content
    post.assert_not_called()


def test_missing_directory_and_symlink_selection_files_refuse(tmp_path):
    regular = tmp_path / "regular"
    regular.write_text("{}")
    link = tmp_path / "link"
    link.symlink_to(regular)
    for path in (tmp_path / "missing", tmp_path, link):
        with patch("httpx.post") as post:
            result = runner.invoke(app, args("run", path))
        assert result.exit_code == 1
        post.assert_not_called()


@pytest.mark.parametrize("status", [403, 404, 409, 413, 415, 422, 503])
def test_private_server_errors_never_retry_or_fetch_fallback(tmp_path, status):
    rid, _, path, _ = prepared(tmp_path)
    with (
        patch("httpx.post", return_value=response({"detail": "private-marker"}, status)) as post,
        patch("httpx.get") as get,
    ):
        result = runner.invoke(app, args(rid, path))
    assert result.exit_code == 1 and "private-marker" not in result.output
    assert post.call_count == 1
    get.assert_not_called()


def test_private_transport_timeout_is_not_continuation_permission(tmp_path):
    rid, _, path, _ = prepared(tmp_path)
    with (
        patch("httpx.post", side_effect=httpx.ReadTimeout("private-marker")) as post,
        patch("httpx.get") as get,
    ):
        result = runner.invoke(app, args(rid, path))
    assert result.exit_code == 1 and "private-marker" not in result.output
    assert "no continuation was submitted" in result.output and post.call_count == 1
    get.assert_not_called()


def test_actual_cli_preview_after_api_restart_preserves_source_and_run_identity(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    provider = ScriptedProvider()
    with TestClient(make_app(tmp_path, provider, max_tool_iterations=1)) as api:
        sid, rid, request = seed(api, root, provider)
        preview = api.post(f"/runs/{rid}/recovery-context", json=request).json()
        run = api.get(f"/runs/{rid}", params={"user_id": "dev"}).json()
        archive = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
    request.pop("user_id")
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(request))
    (root / "a.txt").write_text("later user work\n")
    root.rename(tmp_path / "moved")
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as api:

        def post(url, **kwargs):
            assert kwargs["timeout"] == 30
            assert url.endswith(f"/runs/{rid}/recovery-context")
            return api.post(url.removeprefix("http://localhost:8000"), json=kwargs["json"])

        with patch("httpx.post", side_effect=post) as posts, patch("httpx.get") as gets:
            human = runner.invoke(app, args(rid, path))
            raw = runner.invoke(app, [*args(rid, path), "--json"])
            hidden = runner.invoke(app, [*args(rid, path), "--user-id", "other"])
        assert human.exit_code == raw.exit_code == 0, human.output
        assert hidden.exit_code == 1 and "saved_span" not in hidden.output
        assert "saved\\r\\n" not in hidden.output
        assert json.loads(raw.output) == preview
        assert posts.call_count == 3
        gets.assert_not_called()
        assert api.get(f"/runs/{rid}", params={"user_id": "dev"}).json() == run
        assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archive
        assert len(api.get("/runs", params={"user_id": "dev"}).json()) == 1
    assert (tmp_path / "moved/a.txt").read_text() == "later user work\n"
    assert restarted.calls == [] and len(provider.calls) == 1


@pytest.mark.parametrize("field,value", [("mode", 0o755), ("file_bytes", 100), ("total_lines", 4)])
def test_same_file_ranges_cannot_claim_different_metadata(tmp_path, field, value):
    rid, _, path, data = prepared(tmp_path)
    data["spans"][1][field] = value
    rehash(data)
    with patch("httpx.post", return_value=response(data)):
        result = runner.invoke(app, [*args(rid, path), "--json"])
    assert result.exit_code == 1 and "saved_span" not in result.output


def test_exact_selection_file_byte_limit_accepts_and_overflow_refuses(tmp_path):
    rid, _, path, data = prepared(tmp_path)
    raw = path.read_bytes()
    path.write_bytes(raw + b" " * (8192 - len(raw)))
    with patch("httpx.post", return_value=response(data)) as post:
        result = runner.invoke(app, args(rid, path))
        assert result.exit_code == 0, result.output
        assert post.call_count == 1
        path.write_bytes(path.read_bytes() + b" ")
        overflow = runner.invoke(app, args(rid, path))
        assert overflow.exit_code == 1 and post.call_count == 1
