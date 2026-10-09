from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from harness.config import Settings
from harness.runtime import build_registry, run_configured_turn
from harness.state import Session
from harness.trace import TurnTraceRecord
from memory import FactStore
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tools.process import ProcessResult

ARGV = ["python", "-m", "pytest", "-q", "test_ok.py"]
BAD = '["python", "-m", "pytest", "-q", "test_ok.py"]'


def call(argv: object, identifier: str) -> ToolCall:
    return ToolCall(id=identifier, name="run_command", arguments={"argv": argv})


@pytest.mark.parametrize(
    "case",
    [
        "enabled",
        "default",
        "no_retry",
        "last_iteration",
        "tool_limit",
        "repeated",
        "blank_after",
        "token_limit",
        "truncated",
        "batch_corrected",
        "prior_blank",
    ],
)
async def test_early_recovery_preserves_validation_and_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    process = AsyncMock(return_value=ProcessResult(ARGV, 0, "passed", "", False, False))
    monkeypatch.setattr("tools.code.run_process", process)
    provider = ScriptedProvider()
    first_calls = [call(BAD, "bad")]
    if case == "batch_corrected":
        first_calls.append(call(ARGV, "corrected-in-batch"))
    provider.script(
        *([make_response(content=" ")] if case == "prior_blank" else []),
        make_response(
            tool_calls=first_calls,
            finish_reason="length" if case == "truncated" else "tool_use",
            prompt_tokens=1,
            completion_tokens=1,
        ),
        make_response(
            content=" " if case == "blank_after" else "",
            tool_calls=[]
            if case in ("blank_after", "batch_corrected")
            else [call(BAD if case == "repeated" else ARGV, "retry")],
            prompt_tokens=1,
            completion_tokens=1,
        ),
        make_response(tool_calls=[call(BAD, "blocked")])
        if case == "repeated"
        else make_response(content="Observed result.", prompt_tokens=1, completion_tokens=1),
    )
    settings = Settings(
        _env_file=None,
        recover_string_argv=case != "default",
        project_check_argv=ARGV,
        max_completion_retries=0 if case == "no_retry" else 1,
        max_tool_iterations=1 if case == "last_iteration" else 8,
        max_tool_calls_per_turn=1 if case == "tool_limit" else 24,
        max_completion_tokens_per_turn=1 if case == "token_limit" else 0,
    )
    trace: list[TurnTraceRecord] = []
    with FactStore(tmp_path / "memory.db") as store:
        response = await run_configured_turn(
            settings=settings,
            session=Session(workspace_root=str(tmp_path)),
            user_id="dev",
            message="Run the configured check",
            provider=provider,
            fact_store=store,
            registry=build_registry(fact_store=store, user_id="dev", workspace_root=str(tmp_path)),
            trace=trace,
        )
    early = [
        r for r in trace if r.kind == "recovery" and "rejected a string argv" in r.message.content
    ]
    assert len(early) == (case in ("enabled", "tool_limit", "repeated", "blank_after"))
    assert process.await_count == (
        case in ("enabled", "default", "no_retry", "batch_corrected", "prior_blank")
    )
    if early:
        assert '"argv": ["python", "-m", "pytest", "-q", "test_ok.py"]' in early[0].message.content
        assert provider.calls[1][0][-1] == early[0].message
    if case in ("enabled", "default", "no_retry"):
        assert response.completion_status == "completed"
        assert response.tool_calls[0].arguments["argv"] == BAD
        assert "Resend argv" in response.tool_calls[0].error
        assert response.tool_calls[1].arguments["argv"] == ARGV
    elif case == "repeated":
        assert response.completion_status == "blocked"
        assert sum(r.kind == "recovery" for r in trace) == 1
    elif case == "blank_after":
        assert response.completion_status == "incomplete"
        assert len(provider.calls) == 2  # The early prompt used the shared allowance.
    elif case == "truncated":
        assert response.completion_status == "incomplete"
        assert response.tool_calls == []
    elif case in ("last_iteration", "tool_limit", "token_limit"):
        assert response.completion_status == "budget_exhausted"


async def test_early_prompt_consumes_verification_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = AsyncMock()
    monkeypatch.setattr("tools.code.run_process", process)
    provider = ScriptedProvider()
    provider.script(
        make_response(tool_calls=[call(BAD, "bad")]),
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit", name="write_file", arguments={"path": "a.py", "content": "a=1\n"}
                )
            ]
        ),
        make_response(content="Done."),
    )
    trace: list[TurnTraceRecord] = []
    with FactStore(tmp_path / "memory.db") as store:
        response = await run_configured_turn(
            settings=Settings(
                _env_file=None,
                recover_string_argv=True,
                require_verification_before_finish=True,
                project_check_argv=ARGV,
            ),
            session=Session(workspace_root=str(tmp_path)),
            user_id="dev",
            message="Fix a.py",
            provider=provider,
            fact_store=store,
            registry=build_registry(fact_store=store, user_id="dev", workspace_root=str(tmp_path)),
            trace=trace,
        )
    assert (tmp_path / "a.py").read_text() == "a=1\n"
    assert response.completion_status == "incomplete"
    assert "not verified" in response.completion_reason
    assert sum(r.kind == "recovery" for r in trace) == 1
    process.assert_not_awaited()
