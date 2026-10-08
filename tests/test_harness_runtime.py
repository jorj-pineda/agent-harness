from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.config import Settings
from harness.loop import MAX_ITERATIONS_STUB
from harness.prompts import BASE_SYSTEM_PROMPT
from harness.runtime import build_registry, run_configured_turn
from harness.state import Session, TurnResponse
from memory import FactStore
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tools import ToolError


def test_project_check_settings_accept_json_env_and_reject_non_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROJECT_CHECK_ARGV", '["pytest", "-q"]')
    assert Settings(_env_file=None).project_check_argv == ["pytest", "-q"]
    monkeypatch.delenv("PROJECT_CHECK_ARGV")
    alias = ["python3", "-m", "pytest", "-q"]
    assert Settings(_env_file=None, project_check_argv=alias).project_check_argv == alias
    for argv in ([], ["pytest", "--version"], ["bash", "-c", "pytest"]):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, project_check_argv=argv)


async def test_configured_check_is_shown_in_shared_system_prompt(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, project_check_argv=["pytest", "-q", "test_calc.py"])
    provider = ScriptedProvider()
    provider.script(make_response(content="hello"))
    with FactStore(tmp_path / "memory.db") as store:
        await run_configured_turn(
            settings=settings,
            session=Session(workspace_root=str(tmp_path)),
            user_id="dev",
            message="Inspect the code",
            provider=provider,
            fact_store=store,
            registry=build_registry(fact_store=store, user_id="dev", workspace_root=str(tmp_path)),
        )
    assert '["pytest", "-q", "test_calc.py"]' in provider.calls[0][0][0].content


async def test_runtime_refreshes_memory_without_duplicating_system_message(tmp_path: Path) -> None:
    settings = Settings(_env_file=None)
    session = Session(user_id="dev", workspace_root=str(tmp_path))
    provider = ScriptedProvider()
    provider.script(make_response(content="first"), make_response(content="second"))
    with FactStore(tmp_path / "memory.db") as store:
        registry = build_registry(fact_store=store, user_id="dev", workspace_root=str(tmp_path))
        for message in ("Inspect the code", "Continue"):
            await run_configured_turn(
                settings=settings,
                session=session,
                user_id="dev",
                message=message,
                provider=provider,
                fact_store=store,
                registry=registry,
            )
            store.add("dev", "Use focused changes")
    first, second = [call[0] for call in provider.calls]
    assert first[0].content.startswith(BASE_SYSTEM_PROMPT)
    assert "Use focused changes" not in first[0].content
    assert "Use focused changes" in second[0].content
    assert f"Workspace root: {tmp_path}" in second[0].content
    assert sum(message.role == "system" for message in second) == 1
    assert [message.role for message in second] == ["system", "user", "assistant", "user"]


@pytest.mark.parametrize(
    "policy",
    [
        {"require_plan_before_edit": True},
        {"require_verification_before_finish": True},
        {"max_files_touched_per_turn": 1},
    ],
)
async def test_runtime_applies_configured_policy(tmp_path: Path, policy: dict) -> None:
    settings = Settings(_env_file=None, **policy)
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(id="a", name="write_file", arguments={"path": "a.py", "content": "a=1"}),
                ToolCall(id="b", name="write_file", arguments={"path": "b.py", "content": "b=2"}),
            ]
        ),
        make_response(content="Done"),
        make_response(content="Still done"),
    )
    with FactStore(tmp_path / "memory.db") as store:
        response = await run_configured_turn(
            settings=settings,
            session=Session(workspace_root=str(tmp_path)),
            user_id="dev",
            message="Add two modules",
            provider=provider,
            fact_store=store,
            registry=build_registry(fact_store=store, user_id="dev", workspace_root=str(tmp_path)),
        )
    assert response.escalated is True
    assert response.completion_status == "incomplete"
    if policy.get("require_plan_before_edit"):
        assert response.files_touched == []
        assert not (tmp_path / "a.py").exists()
        assert not (tmp_path / "b.py").exists()
    elif policy.get("max_files_touched_per_turn"):
        assert response.files_touched == ["a.py"]
        assert (tmp_path / "a.py").exists()
        assert not (tmp_path / "b.py").exists()
    else:
        assert response.files_touched == ["a.py", "b.py"]
        assert response.verification_status == "not_run"


async def test_runtime_uses_iteration_budget_and_scope_policy(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    provider.script(make_response(tool_calls=[ToolCall(id="a", name="recall", arguments={})]))
    with FactStore(tmp_path / "memory.db") as store:
        registry = build_registry(fact_store=store, user_id="dev", workspace_root=None)
        kwargs = {
            "settings": Settings(_env_file=None, max_tool_iterations=1),
            "session": Session(),
            "user_id": "dev",
            "provider": provider,
            "fact_store": store,
            "registry": registry,
        }
        refused = await run_configured_turn(message="delete .git", **kwargs)
        assert refused.provider == "policy"
        assert refused.completion_status == "blocked"
        assert provider.calls == []
        response = await run_configured_turn(message="Recall my preferences", **kwargs)
    assert response.answer == MAX_ITERATIONS_STUB
    assert len(provider.calls) == 1


async def test_runtime_applies_output_budget_when_provider_omits_usage(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "a.py", "content": "a=1"},
                )
            ]
        )
    )
    with FactStore(tmp_path / "memory.db") as store:
        response = await run_configured_turn(
            settings=Settings(_env_file=None, max_completion_tokens_per_turn=5),
            session=Session(workspace_root=str(tmp_path)),
            user_id="dev",
            message="Edit a.py",
            provider=provider,
            fact_store=store,
            registry=build_registry(fact_store=store, user_id="dev", workspace_root=str(tmp_path)),
        )
    assert response.completion_status == "incomplete"
    assert "did not report" in (response.completion_reason or "")
    assert not (tmp_path / "a.py").exists()


async def _run_edit_turn(
    workspace: Path, memory: Path, settings: Settings, *calls: ToolCall
) -> TurnResponse:
    provider = ScriptedProvider()
    provider.script(make_response(tool_calls=list(calls)), make_response(content="Done"))
    with FactStore(memory / "memory.db") as store:
        return await run_configured_turn(
            settings=settings,
            session=Session(workspace_root=str(workspace)),
            user_id="dev",
            message="Fix the module",
            provider=provider,
            fact_store=store,
            registry=build_registry(
                fact_store=store,
                user_id="dev",
                workspace_root=str(workspace),
                command_env={
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            ),
        )


async def test_runtime_reports_turn_changes_and_excludes_preexisting_edits(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "user_dirty.py").write_text("# user edit before the turn\n", encoding="utf-8")
    (workspace / "remove_me.txt").write_text("x", encoding="utf-8")
    (workspace / "test_side_effect.py").write_text(
        "from pathlib import Path\n"
        "def test_side_effect():\n"
        "    Path('remove_me.txt').unlink()\n"
        "    Path('generated.txt').write_text('from pytest')\n",
        encoding="utf-8",
    )
    response = await _run_edit_turn(
        workspace,
        tmp_path,
        Settings(_env_file=None),
        ToolCall(id="w", name="write_file", arguments={"path": "a.py", "content": "a = 1\n"}),
        ToolCall(id="t", name="run_command", arguments={"argv": ["pytest", "-q"]}),
    )
    changes = response.workspace_changes
    assert changes.status == "tracked"
    assert changes.added == ["a.py", "generated.txt"]
    assert changes.modified == []
    assert changes.deleted == ["remove_me.txt"]
    assert response.files_touched == ["a.py"]
    assert (workspace / "user_dirty.py").read_text(encoding="utf-8").startswith("# user edit")


async def test_runtime_reports_unavailable_changes_over_limit(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "one.py").write_text("", encoding="utf-8")
    (workspace / "two.py").write_text("", encoding="utf-8")
    edit = ToolCall(id="w", name="write_file", arguments={"path": "a.py", "content": "a = 1\n"})
    limited = await _run_edit_turn(
        workspace, tmp_path, Settings(_env_file=None, max_tracked_files=1), edit
    )
    assert limited.workspace_changes.status == "unavailable"
    assert "more than 1 tracked files" in (limited.workspace_changes.reason or "")
    assert limited.workspace_changes.added == []
    assert limited.files_touched == ["a.py"]

    disabled = await _run_edit_turn(
        workspace, tmp_path, Settings(_env_file=None, track_workspace_changes=False), edit
    )
    assert disabled.workspace_changes.status == "not_tracked"


async def test_runtime_applies_context_limit_before_provider_request(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    with FactStore(tmp_path / "memory.db") as store:
        response = await run_configured_turn(
            settings=Settings(_env_file=None, max_context_tokens=500),
            session=Session(),
            user_id="dev",
            message="Explain the module",
            provider=provider,
            fact_store=store,
            registry=build_registry(fact_store=store, user_id="dev", workspace_root=None),
        )
    assert provider.calls == []
    assert response.completion_status == "budget_exhausted"
    assert "context limit" in (response.completion_reason or "")


async def test_runtime_bounds_blank_final_recovery(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    provider.script(make_response(content=""), make_response(content=" \n"))
    with FactStore(tmp_path / "memory.db") as store:
        response = await run_configured_turn(
            settings=Settings(_env_file=None, max_tool_iterations=3, max_completion_retries=1),
            session=Session(),
            user_id="dev",
            message="Explain the module",
            provider=provider,
            fact_store=store,
            registry=build_registry(fact_store=store, user_id="dev", workspace_root=None),
        )
    assert len(provider.calls) == 2
    assert response.completion_status == "incomplete"
    assert "empty final answer" in (response.completion_reason or "")
    assert response.answer.strip()


def test_toolset_setting_validates_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODING_TOOLSET", "whole_file")
    assert Settings(_env_file=None).coding_toolset == "whole_file"
    monkeypatch.setenv("CODING_TOOLSET", "unknown")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


async def test_whole_file_toolset_rejects_exact_edits_and_preserves_other_specs(
    tmp_path: Path,
) -> None:
    target = tmp_path / "a.py"
    target.write_text("original\n", encoding="utf-8")
    with FactStore(tmp_path / "memory.db") as store:
        full = build_registry(fact_store=store, user_id="dev", workspace_root=str(tmp_path))
        reduced = build_registry(
            fact_store=store,
            user_id="dev",
            workspace_root=str(tmp_path),
            coding_toolset="whole_file",
        )
        assert reduced.as_tool_specs() == [
            spec for spec in full.as_tool_specs() if spec.name != "replace_text"
        ]
        with pytest.raises(ToolError, match="Unknown tool"):
            await reduced.invoke("replace_text", {"path": "a.py"})
        assert target.read_text(encoding="utf-8") == "original\n"
        await reduced.invoke("write_file", {"path": "a.py", "content": "changed\n"})
        assert target.read_text(encoding="utf-8") == "changed\n"


@pytest.mark.parametrize("mode", ["corrected", "repeated", "tool_limit"])
async def test_string_command_recovery_uses_shared_feedback_and_existing_budgets(
    tmp_path: Path, mode: str
) -> None:
    (tmp_path / "test_ok.py").write_text(
        "from pathlib import Path\ndef test_ok():\n    Path('command-ran').write_text('yes')\n",
        encoding="utf-8",
    )
    argv = ["python", "-m", "pytest", "-q", "test_ok.py"]
    malformed = '["python", "-m", "pytest", "-q", "test_ok.py"]'
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[ToolCall(id="bad", name="run_command", arguments={"argv": malformed})]
        ),
        make_response(
            tool_calls=[
                ToolCall(
                    id="retry",
                    name="run_command",
                    arguments={"argv": malformed if mode == "repeated" else argv},
                )
            ]
        ),
        make_response(
            tool_calls=[ToolCall(id="again", name="run_command", arguments={"argv": malformed})]
        )
        if mode == "repeated"
        else make_response(content="Check passed."),
    )
    settings = Settings(
        _env_file=None,
        project_check_argv=argv,
        max_tool_calls_per_turn=1 if mode == "tool_limit" else 24,
    )
    with FactStore(tmp_path / "memory.db") as store:
        response = await run_configured_turn(
            settings=settings,
            session=Session(workspace_root=str(tmp_path)),
            user_id="dev",
            message="Run the configured check",
            provider=provider,
            fact_store=store,
            registry=build_registry(
                fact_store=store,
                user_id="dev",
                workspace_root=str(tmp_path),
                command_env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            ),
        )
    assert response.tool_calls[0].arguments == {"argv": malformed}
    assert "Resend argv as a JSON array" in (response.tool_calls[0].error or "")
    tool_messages = [m for m in provider.calls[1][0] if m.role == "tool"]
    assert "Resend argv as a JSON array" in tool_messages[-1].content
    assert (tmp_path / "command-ran").exists() == (mode == "corrected")
    if mode == "corrected":
        assert response.completion_status == "completed"
        assert response.verification_status == "passed"
        assert response.tool_calls[1].arguments == {"argv": argv}
    else:
        assert response.completion_status == (
            "blocked" if mode == "repeated" else "budget_exhausted"
        )
        assert all(call.result is None for call in response.tool_calls)
