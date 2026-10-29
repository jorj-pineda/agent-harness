"""Thin prospective review with explicit hypothetical budgets and no implicit execution."""

import hashlib
import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli.main import app
from memory import FactStore
from tests.api.conftest import ScriptedProvider
from tests.api.test_recovery_context import seed
from tests.api.test_session_persistence import make_app
from tests.test_recovery_draft import draft, prepared

runner = CliRunner()


def fixture(tmp_path, **options):
    root, store, record, archive, request = prepared(tmp_path)
    archive.session.messages[1].content += "\x1b[2J"
    archive.session.turns[0].user_input += "\x1b[2J"
    if options.pop("with_instructions", False):
        (root / "AGENTS.md").write_bytes(b"Current conventions\r\n")
        options["load_root_agents"] = True
    with FactStore(tmp_path / "facts.db") as facts:
        result = draft(store, record, archive, request, facts, **options)
    store.close()
    selection = request.model_dump(mode="json", exclude={"message"})
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    return record.run_id, selection, request.message, path, result.model_dump(mode="json")


def args(rid, message, path):
    return [
        "recovery-draft",
        rid,
        "--selection-file",
        str(path),
        "--message",
        message,
        "--user-id",
        "dev",
    ]


def response(value, status=200):
    return httpx.Response(status, json=value, request=httpx.Request("POST", "http://fixture"))


def rehash(data):
    def compact(item):
        return json.dumps(item, ensure_ascii=False, separators=(",", ":"))

    data["messages_sha256"] = hashlib.sha256(
        "\n".join(compact(m) for m in data["messages"]).encode()
    ).hexdigest()


def test_human_and_json_retain_full_draft_scope_and_escape_sensitive_text(tmp_path):
    rid, selection, message, path, data = fixture(tmp_path)
    before = path.read_bytes()
    with patch("httpx.post", return_value=response(data)) as post, patch("httpx.get") as get:
        human = runner.invoke(app, [*args(rid, message, path), "--provider", "ollama"])
        raw = runner.invoke(app, [*args(rid, message, path), "--provider", "ollama", "--json"])
    assert human.exit_code == raw.exit_code == 0, human.output
    assert "execution_supported=false" in human.output
    assert "Estimated fit does not authorize execution" in human.output
    assert "hypothetical-new-turn" in human.output and "no source allowance reset" in human.output
    assert '"status": "estimated_fit"' in human.output
    assert (
        "retained_user_message_count=3" in human.output
        and "omitted_nonuser_message_count=5" in human.output
    )
    assert "not measured provider tokens or savings" in human.output
    assert "\\u001b" in human.output and "\x1b" not in human.output
    assert json.loads(raw.output) == data
    post.assert_called_with(
        f"http://localhost:8000/runs/{rid}/recovery-draft",
        json={"user_id": "dev", **selection, "message": message, "provider": "ollama"},
        timeout=30,
    )
    assert post.call_count == 2 and path.read_bytes() == before
    get.assert_not_called()


@pytest.mark.parametrize(
    "options,status", [({"max_total_tokens_per_turn": 1}, "blocked"), ({}, "estimated_fit")]
)
def test_reported_context_states_never_enable_execution(tmp_path, options, status):
    rid, _, message, path, data = fixture(tmp_path, **options)
    with patch("httpx.post", return_value=response(data)) as post:
        result = runner.invoke(app, args(rid, message, path))
    assert result.exit_code == 0, result.output
    assert f'"status": "{status}"' in result.output
    assert "execution_supported=false" in result.output and post.call_count == 1


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "draftversion",
        "run",
        "runhash",
        "archivehash",
        "evidencehash",
        "messageshash",
        "toolhash",
        "budgetscope",
        "readonly",
        "execute",
        "size",
        "provider",
        "countbool",
        "usercount",
        "omitted",
        "evidenceindex",
        "messagecount",
        "role",
        "toolcall",
        "continuation",
        "proposed",
        "userbytes",
        "prefix",
        "embedded",
        "estimatebool",
        "ratio",
        "fit",
        "blocked",
        "unbounded",
        "configuration",
        "configurationcount",
        "instructionstatus",
        "instructionhash",
        "wire",
    ],
)
@pytest.mark.parametrize("raw", [False, True])
def test_inconsistent_drafts_refuse_before_either_output_mode(tmp_path, change, raw):
    rid, _, message, path, data = fixture(tmp_path)
    if change == "version":
        data["schema_version"] = True
    elif change == "draftversion":
        data["draft_version"] = "future"
    elif change == "run":
        data["run_id"] = "other"
    elif change == "runhash":
        data["source_record_sha256"] = "0" * 64
    elif change == "archivehash":
        data["source_archive_sha256"] = "bad"
    elif change == "evidencehash":
        data["recovery_evidence_sha256"] = "0" * 64
    elif change == "messageshash":
        data["messages_sha256"] = "0" * 64
    elif change == "toolhash":
        data["tool_specs_sha256"] = "bad"
    elif change == "budgetscope":
        data["budget_scope"] = "source-allowance-reset"
    elif change == "readonly":
        data["read_only"] = False
    elif change == "execute":
        data["execution_supported"] = 0
    elif change == "size":
        data["max_serialized_bytes"] = True
    elif change == "provider":
        data["provider"] = "other"
    elif change == "countbool":
        data["source_message_count"] = True
    elif change == "usercount":
        data["retained_user_message_count"] = 65
    elif change == "omitted":
        data["omitted_nonuser_message_count"] += 1
    elif change == "evidenceindex":
        data["evidence_message_index"] = 1
    elif change == "messagecount":
        data["messages"].pop(1)
    elif change == "role":
        data["messages"][1]["role"] = "assistant"
    elif change == "toolcall":
        data["messages"][1]["tool_calls"] = [{"name": "write_file"}]
    elif change == "continuation":
        data["messages"][1]["continuation"] = {"text": "private-marker"}
    elif change == "proposed":
        data["messages"][-1]["content"] = "other"
    elif change == "userbytes":
        data["messages"][1]["content"] = "a" * 32769
    elif change == "prefix":
        data["messages"][data["evidence_message_index"]]["content"] = "No historical label"
    elif change == "embedded":
        index = data["evidence_message_index"]
        prefix, packet = data["messages"][index]["content"].split("\n", 1)
        evidence = json.loads(packet)
        evidence["review"]["run_id"] = "other"
        data["messages"][index]["content"] = (
            prefix + "\n" + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
        )
    elif change == "estimatebool":
        data["context"]["estimated_prompt_tokens"] = True
    elif change == "ratio":
        data["context"]["prompt_estimate_bytes_per_token"] = 0.5
    elif change == "fit":
        data["context"]["max_context_tokens"] = 0
    elif change == "blocked":
        data["context"]["status"] = "blocked"
    elif change == "unbounded":
        data["context"]["status"] = "unbounded_context"
    elif change == "configuration":
        data["configuration"]["status"] = "matching"
        data["configuration"]["unknown_fields"] = ["model"]
    elif change == "configurationcount":
        data["configuration"]["changed_fields"] = ["prompt"]
    elif change == "instructionstatus":
        data["repository_instructions"]["status"] = "unavailable"
    elif change == "instructionhash":
        data["repository_instructions"]["content_sha256"] = "0" * 64
    elif change == "wire":
        data["extra"] = "private-marker" * 131072
    if change != "messageshash":
        rehash(data)
    with patch("httpx.post", return_value=response(data)) as post, patch("httpx.get") as get:
        result = runner.invoke(
            app, [*args(rid, message, path), "--provider", "ollama", *(["--json"] if raw else [])]
        )
    assert result.exit_code == 1 and "prospective_recovery_draft" not in result.output
    assert "private-marker" not in result.output and "Previous agreement" not in result.output
    assert "No automatic retry" in result.output
    assert post.call_count == 1
    get.assert_not_called()


@pytest.mark.parametrize("message", ["", " ", "🙂" * 1025])
def test_invalid_proposed_message_does_not_dispatch(tmp_path, message):
    rid, _, _, path, _ = fixture(tmp_path)
    with patch("httpx.post") as post:
        result = runner.invoke(app, args(rid, message, path))
    assert result.exit_code == 1
    post.assert_not_called()


@pytest.mark.parametrize("content", [b"[]", b"x" * 8193, b"\xff", b'{"spans":[],"spans":[]}'])
def test_invalid_selection_files_refuse_without_request(tmp_path, content):
    path = tmp_path / "selection.json"
    path.write_bytes(content)
    with patch("httpx.post") as post:
        result = runner.invoke(app, args("run", "Review", path))
    assert result.exit_code == 1
    post.assert_not_called()


@pytest.mark.parametrize("status", [400, 403, 404, 409, 413, 415, 422, 503])
def test_private_api_refusals_never_retry_or_fetch_history(tmp_path, status):
    rid, _, message, path, _ = fixture(tmp_path)
    with (
        patch("httpx.post", return_value=response({"detail": "private-marker"}, status)) as post,
        patch("httpx.get") as get,
    ):
        result = runner.invoke(app, args(rid, message, path))
    assert result.exit_code == 1 and "private-marker" not in result.output
    assert "No continuation was submitted" in result.output and post.call_count == 1
    get.assert_not_called()


def test_private_timeout_does_not_dispatch_a_second_request(tmp_path):
    rid, _, message, path, _ = fixture(tmp_path)
    with (
        patch("httpx.post", side_effect=httpx.ReadTimeout("private-marker")) as post,
        patch("httpx.get") as get,
    ):
        result = runner.invoke(app, args(rid, message, path))
    assert result.exit_code == 1 and "private-marker" not in result.output and post.call_count == 1
    get.assert_not_called()


def test_cli_to_restarted_api_retains_draft_source_identity_and_later_user_work(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    provider = ScriptedProvider()
    with TestClient(
        make_app(tmp_path, provider, max_tool_iterations=1, max_context_tokens=16384)
    ) as api:
        sid, rid, body = seed(api, root, provider, known_usage=True)
        message = "Finish the original task"
        expected = api.post(f"/runs/{rid}/recovery-draft", json={**body, "message": message}).json()
        run = api.get(f"/runs/{rid}", params={"user_id": "dev"}).json()
        archive = api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json()
    body.pop("user_id")
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(body))
    (root / "a.txt").write_text("later user work\n")
    restarted = ScriptedProvider()
    with TestClient(
        make_app(tmp_path, restarted, max_tool_iterations=1, max_context_tokens=16384)
    ) as api:

        def post(url, **kwargs):
            assert kwargs["timeout"] == 30 and url.endswith(f"/runs/{rid}/recovery-draft")
            return api.post(url.removeprefix("http://localhost:8000"), json=kwargs["json"])

        with patch("httpx.post", side_effect=post) as posts, patch("httpx.get") as gets:
            human = runner.invoke(app, args(rid, message, path))
            raw = runner.invoke(app, [*args(rid, message, path), "--json"])
            hidden = runner.invoke(app, [*args(rid, message, path), "--user-id", "other"])
        assert human.exit_code == raw.exit_code == 0, human.output
        assert hidden.exit_code == 1 and "prospective_recovery_draft" not in hidden.output
        assert json.loads(raw.output) == expected
        assert posts.call_count == 3
        gets.assert_not_called()
        assert api.get(f"/runs/{rid}", params={"user_id": "dev"}).json() == run
        assert api.get(f"/sessions/{sid}", params={"user_id": "dev"}).json() == archive
        assert len(api.get("/runs", params={"user_id": "dev"}).json()) == 1
    assert (root / "a.txt").read_text() == "later user work\n"
    assert restarted.calls == [] and len(provider.calls) == 1


def test_unbounded_context_is_rendered_without_fit_or_permission_claim(tmp_path):
    rid, _, message, path, data = fixture(tmp_path)
    data["context"].update(
        status="unbounded_context",
        max_context_tokens=0,
        configured_max_context_tokens=0,
        request_output_allowance=None,
    )
    with patch("httpx.post", return_value=response(data)):
        result = runner.invoke(app, args(rid, message, path))
    assert result.exit_code == 0, result.output
    assert '"status": "unbounded_context"' in result.output
    assert "execution_supported=false" in result.output
    assert 'source_completion="budget_exhausted" source_checks="failed"' in result.output


def test_loaded_current_instruction_identity_remains_visible(tmp_path):
    rid, _, message, path, data = fixture(tmp_path, with_instructions=True)
    with patch("httpx.post", return_value=response(data)):
        result = runner.invoke(app, [*args(rid, message, path), "--json"])
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["repository_instructions"]["status"] == "loaded"
    assert value["repository_instructions"]["content_bytes"] == len(b"Current conventions\r\n")
    assert "Current conventions" in value["messages"][0]["content"]


@pytest.mark.parametrize("field", ["completion_status", "verification_status"])
def test_missing_source_status_refuses_even_with_consistent_all_message_hashes(tmp_path, field):
    rid, _, message, path, data = fixture(tmp_path)
    index = data["evidence_message_index"]
    prefix, packet = data["messages"][index]["content"].split("\n", 1)
    evidence = json.loads(packet)
    evidence["review"].pop(field)

    def compact(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    joined = compact(evidence["review"]) + "\n" + "\n".join(compact(s) for s in evidence["spans"])
    evidence["evidence_sha256"] = hashlib.sha256(joined.encode()).hexdigest()
    data["recovery_evidence_sha256"] = evidence["evidence_sha256"]
    data["messages"][index]["content"] = prefix + "\n" + compact(evidence)
    rehash(data)
    with patch("httpx.post", return_value=response(data)):
        result = runner.invoke(app, [*args(rid, message, path), "--json"])
    assert result.exit_code == 1 and "messages" not in result.output
