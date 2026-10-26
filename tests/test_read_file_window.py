from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from evals.real_run import load_tasks, run_task
from harness.config import Settings
from harness.configuration_review import compare_profiles
from harness.execution_profile import ExecutionProfile, capture_execution_profile
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tests.api.test_session_persistence import make_app
from tools import ToolError, ToolRegistry
from tools.code import build_code_tools
from workspace import Workspace


def registry(root: Path, lines: int) -> ToolRegistry:
    result = ToolRegistry()
    for tool in build_code_tools(Workspace(root=root), read_file_default_lines=lines):
        result.register(tool)
    return result


async def test_window_pages_exact_crlf_unicode_with_full_hash_and_explicit_override(tmp_path):
    raw = "one\r\n雪\r\nthree\nfour".encode()
    target = tmp_path / "a.py"
    target.write_bytes(raw)
    tools = registry(tmp_path, 2)
    first = await tools.invoke("read_file", {"path": "a.py"})
    assert first == {
        "path": "a.py",
        "start_line": 1,
        "end_line": 2,
        "content": "one\r\n雪",
        "total_lines": 4,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "read_window": {"default_lines": 2, "has_more": True, "next_start_line": 3},
    }
    last = await tools.invoke("read_file", {"path": "a.py", "start_line": 3})
    assert last["content"] == "three\nfour" and last["end_line"] == 4
    assert last["read_window"] == {"default_lines": 2, "has_more": False, "next_start_line": None}
    assert last["sha256"] == first["sha256"]
    explicit = {"path": "a.py", "start_line": 2, "end_line": 4}
    assert await tools.invoke("read_file", explicit) == await registry(tmp_path, 0).invoke(
        "read_file", explicit
    )
    assert target.read_bytes() == raw
    target.write_bytes(raw + b"\nexternal")
    with pytest.raises(ToolError, match="changed"):
        await tools.invoke(
            "replace_text",
            {
                "path": "a.py",
                "old_text": "one",
                "new_text": "edited",
                "expected_sha256": first["sha256"],
            },
        )
    assert target.read_bytes() == raw + b"\nexternal"


async def test_default_zero_retains_legacy_result_and_window_does_not_hide_limits(
    tmp_path, monkeypatch
):
    (tmp_path / "a.py").write_bytes(b"line\n" * 600)
    tools = registry(tmp_path, 2)
    read = await tools.invoke("read_file", {"path": "a.py"})
    assert read["total_lines"] == 600 and read["end_line"] == 2
    with pytest.raises(ToolError, match="Line range exceeds"):
        await registry(tmp_path, 0).invoke("read_file", {"path": "a.py"})
    with pytest.raises(ToolError, match="Line range exceeds"):
        await tools.invoke("read_file", {"path": "a.py", "end_line": 501})
    monkeypatch.setattr("tools.code.MAX_READ_SCAN_BYTES", 20)
    with pytest.raises(ToolError, match="whole-file scan exceeds"):
        await tools.invoke("read_file", {"path": "a.py"})
    (tmp_path / "long.py").write_bytes(b"0123456789\n")
    monkeypatch.setattr("tools.code.MAX_READ_BYTES", 5)
    with pytest.raises(ToolError, match="Selected range exceeds"):
        await tools.invoke("read_file", {"path": "long.py"})


async def test_empty_and_past_end_windows_do_not_invent_lines(tmp_path):
    (tmp_path / "empty").write_bytes(b"")
    tools = registry(tmp_path, 1)
    read = await tools.invoke("read_file", {"path": "empty"})
    assert read["end_line"] == read["total_lines"] == 0 and read["content"] == ""
    assert read["read_window"]["next_start_line"] is None
    with pytest.raises(ToolError, match="beyond file length"):
        await tools.invoke("read_file", {"path": "empty", "start_line": 2})


@pytest.mark.parametrize("bad", [-1, 501, True, 1.5])
def test_window_settings_and_direct_builder_reject_invalid_values(tmp_path, bad):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, read_file_default_lines=bad)
    with pytest.raises(ValueError):
        registry(tmp_path, bad)


def test_window_env_and_legacy_profile_default(monkeypatch):
    monkeypatch.setenv("READ_FILE_DEFAULT_LINES", "80")
    assert Settings(_env_file=None).read_file_default_lines == 80
    monkeypatch.delenv("READ_FILE_DEFAULT_LINES")
    first = capture_execution_profile(Settings(_env_file=None), "ollama", ToolRegistry())
    old = first.model_dump()
    old["settings"].pop("read_file_default_lines")
    restored = ExecutionProfile.model_validate(old)
    assert restored.settings.read_file_default_lines == 0
    changed = capture_execution_profile(
        Settings(_env_file=None, read_file_default_lines=80), "ollama", ToolRegistry()
    )
    assert "settings.read_file_default_lines" in compare_profiles(restored, changed).changed_fields


def test_api_selected_window_reaches_real_tool_without_changing_explicit_ranges(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("one\ntwo\nthree\n")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(id="default", name="read_file", arguments={"path": "a.py"}),
                ToolCall(
                    id="explicit",
                    name="read_file",
                    arguments={
                        "path": "a.py",
                        "start_line": 2,
                        "end_line": 3,
                    },
                ),
            ]
        ),
        make_response(content="Inspected."),
    )
    with TestClient(make_app(tmp_path, provider, read_file_default_lines=1)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        result = client.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Inspect a.py"}
        ).json()
        saved = client.get("/runs/" + result["run_id"], params={"user_id": "dev"}).json()
        assert saved["execution_profile"]["settings"]["read_file_default_lines"] == 1
    replies = [json.loads(m.content) for m in provider.calls[-1][0] if m.role == "tool"]
    assert replies[0]["content"] == "one" and replies[0]["read_window"]["has_more"]
    assert replies[1]["content"] == "two\nthree" and "read_window" not in replies[1]
    assert (repo / "a.py").read_text() == "one\ntwo\nthree\n"


async def test_real_evaluator_uses_same_window_and_records_policy():
    task = load_tasks()[0]
    before = (task.fixture / "calc.py").read_bytes()
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(id="read", name="read_file", arguments={"path": "calc.py"}),
            ]
        ),
        make_response(content="Inspected."),
    )
    result = await run_task(
        task,
        provider=provider,
        settings=Settings(
            _env_file=None,
            read_file_default_lines=1,
        ),
        mode="harness",
    )
    assert not result.passed  # An unchanged failing artifact is not accepted.
    assert result.runtime_config["read_file_default_lines"] == 1
    reply = next(m for m in provider.calls[-1][0] if m.role == "tool")
    body = json.loads(reply.content)
    assert body["end_line"] == 1 and body["read_window"]["has_more"]
    assert body["sha256"] == hashlib.sha256(before).hexdigest()
    assert (task.fixture / "calc.py").read_bytes() == before
