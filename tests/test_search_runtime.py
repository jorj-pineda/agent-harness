"""Configured runtime exposes search overflow as a recoverable tool error."""

from pathlib import Path

from harness.config import Settings
from harness.runtime import build_registry, run_configured_turn
from harness.state import Session
from memory import FactStore
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tools.code import MAX_GREP_HITS


async def test_search_overflow_then_narrowing_preserves_real_workspace(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    original = "needle\n" * (MAX_GREP_HITS + 1)
    (root / "many.py").write_text(original)
    (root / "one.py").write_text("needle\n")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="broad",
                    name="grep_repo",
                    arguments={
                        "pattern": "needle",
                    },
                )
            ],
            finish_reason="tool_use",
        ),
        make_response(
            tool_calls=[
                ToolCall(
                    id="narrow",
                    name="grep_repo",
                    arguments={
                        "pattern": "needle",
                        "path": "one.py",
                    },
                )
            ],
            finish_reason="tool_use",
        ),
        make_response(content="one.py has a match; the broad search exceeded its limit."),
    )
    session = Session(user_id="dev", workspace_root=str(root))
    with FactStore(tmp_path / "facts.db") as store:
        response = await run_configured_turn(
            settings=Settings(_env_file=None),
            session=session,
            user_id="dev",
            message="Find needle in the source",
            provider=provider,
            fact_store=store,
            registry=build_registry(fact_store=store, user_id="dev", workspace_root=str(root)),
        )
    assert "results are incomplete" in (response.tool_calls[0].error or "")
    assert response.tool_calls[0].result is None
    assert response.tool_calls[1].error is None
    assert response.tool_calls[1].result == [{"path": "one.py", "line": 1, "text": "needle"}]
    assert "Narrow" in next(
        message.content for message in provider.calls[1][0] if message.role == "tool"
    )
    assert (root / "many.py").read_text() == original
    assert (root / "one.py").read_text() == "needle\n"
    assert response.workspace_changes is not None
    assert response.workspace_changes.status == "tracked"
    assert response.workspace_changes.added == []
    assert response.workspace_changes.modified == []
    assert response.workspace_changes.deleted == []
