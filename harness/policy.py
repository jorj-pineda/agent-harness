"""Lightweight turn policy: scope gate and edit budget checks."""

from __future__ import annotations

from harness.outcome import EMIT_PLAN_TOOL_NAME, WRITE_FILE_TOOL_NAME
from harness.state import ToolCallRecord

OUT_OF_SCOPE_PHRASES = (
    "delete .git",
    "rm -rf",
    "rewrite the entire codebase",
    "refactor every file",
    "exfiltrate",
    "drop database",
)


def is_out_of_scope_request(user_input: str) -> bool:
    """Heuristic scope gate — refuse clearly unsafe or unbounded requests."""
    lowered = user_input.lower()
    return any(phrase in lowered for phrase in OUT_OF_SCOPE_PHRASES)


def edit_budget_exceeded(files_touched: list[str], *, max_files: int) -> bool:
    """True when the turn touched more distinct files than allowed."""
    if max_files < 1:
        return False
    return len(set(files_touched)) > max_files


def edit_without_plan(tool_calls: list[ToolCallRecord]) -> bool:
    """True when a successful write_file ran before any successful emit_plan this turn."""
    saw_plan = False
    for call in tool_calls:
        if call.name == EMIT_PLAN_TOOL_NAME and call.error is None:
            saw_plan = True
        elif call.name == WRITE_FILE_TOOL_NAME and call.error is None and not saw_plan:
            return True
    return False
