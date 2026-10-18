import pytest
from pydantic import ValidationError

from harness.access import INSPECTION_TOOL_NAMES
from harness.config import Settings
from harness.loop import run_turn
from harness.runtime import build_registry, run_configured_turn
from harness.state import Session
from harness.transcript import inspect_transcript
from memory import FactStore
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("write_file", {"path": "user.py", "content": "overwritten"}),
        ("replace_text", {"path": "user.py"}),
        ("run_command", {"argv": ["pytest"]}),
        ("remember_fact", {"fact": "must not persist"}),
        ("emit_plan", {"steps": ["edit"]}),
        ("git_status", {}),
        ("unknown", {}),
    ],
)
async def test_inspection_rejects_entire_mixed_batch_before_dispatch(tmp_path, name, arguments):
    target = tmp_path / "user.py"
    target.write_text("user = 1\n")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(id="allowed", name="read_file", arguments={"path": "user.py"}),
                ToolCall(id="forbidden", name=name, arguments=arguments),
            ]
        )
    )
    session = Session(user_id="dev", workspace_root=str(tmp_path), inspection_only=True)
    with FactStore(tmp_path / "memory.db") as store:
        # Deliberately supply an unrestricted registry to exercise the loop's independent guard.
        registry = build_registry(fact_store=store, user_id="dev", workspace_root=str(tmp_path))
        result = await run_turn(
            session=session, user_input="Inspect", provider=provider, registry=registry
        )
        assert store.list("dev") == []
    assert result.completion_status == "blocked" and result.escalated
    assert "No tools in this batch were run" in result.completion_reason
    assert result.tool_calls == [] and result.files_touched == []
    assert result.verification_status == "not_run" and not result.verification_ran
    assert target.read_text() == "user = 1\n"
    assert {tool.name for tool in provider.calls[0][1]} == INSPECTION_TOOL_NAMES
    assert inspect_transcript(session.messages).status == "valid"


async def test_inspection_shared_runtime_reads_and_answers_without_required_checks(tmp_path):
    target = tmp_path / "user.py"
    target.write_text("user = 1\n")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[ToolCall(id="read", name="read_file", arguments={"path": "user.py"})]
        ),
        make_response(content="The module sets user to 1."),
    )
    session = Session(user_id="dev", workspace_root=str(tmp_path), inspection_only=True)
    profiles = []
    with FactStore(tmp_path / "memory.db") as store:
        registry = build_registry(fact_store=store, user_id="dev", workspace_root=str(tmp_path))
        result = await run_configured_turn(
            settings=Settings(
                _env_file=None,
                require_verification_before_finish=True,
                project_check_argv=["pytest"],
            ),
            session=session,
            user_id="dev",
            message="Explain this module",
            provider=provider,
            fact_store=store,
            registry=registry,
            on_execution_profile=profiles.append,
        )
    assert result.completion_status == "completed" and result.verification_status == "not_run"
    assert result.files_touched == [] and result.memory_writes == []
    assert result.workspace_changes.modified == []
    assert profiles[0].inspection_only is True
    assert set(profiles[0].tool_names) == INSPECTION_TOOL_NAMES
    prompt = provider.calls[0][0][0].content
    assert "Inspection-only session" in prompt and "No project check is required" in prompt
    assert "After editing, run this project check" not in prompt
    assert target.read_text() == "user = 1\n"


def test_inspection_flag_is_strict_fixed_and_legacy_sessions_remain_standard():
    assert Session().inspection_only is False
    for value in ("true", 1, None):
        with pytest.raises(ValidationError):
            Session(inspection_only=value)
    session = Session(inspection_only=True)
    with pytest.raises(ValidationError):
        session.inspection_only = False
    assert session.model_copy(deep=True).inspection_only is True
