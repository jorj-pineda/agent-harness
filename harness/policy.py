"""Lightweight turn policy: scope gate and edit budget checks."""

from __future__ import annotations

from typing import Literal

from harness.outcome import EMIT_PLAN_TOOL_NAME, WRITE_FILE_TOOL_NAME
from harness.state import ToolCallRecord

TaskKind = Literal["bugfix", "explore", "refactor", "out_of_scope"]

OUT_OF_SCOPE_PHRASES = (
    "delete .git",
    "rm -rf",
    "rewrite the entire codebase",
    "rewrite entire repo",
    "rewrite the whole repo",
    "rewrite whole codebase",
    "refactor every file",
    "delete all tests",
    "remove all tests",
    "delete every test",
    "remove every test",
    "exfiltrate",
    "drop database",
)

EXPLORE_SIGNALS = (
    "what ",
    "what's",
    "where ",
    "which ",
    "how does",
    "how do ",
    "explain ",
    "describe ",
    "list ",
    "show me",
    "tell me about",
    "grep ",
    "search for",
    "find ",
    "according to",
)

REFACTOR_SIGNALS = (
    "refactor",
    "rename ",
    "extract ",
    "reorganize",
    "move module",
    "inline ",
)

BUGFIX_SIGNALS = (
    "fix ",
    "fix the",
    "bug",
    "failing",
    "broken",
    "error",
    "patch ",
)


def classify_task(user_input: str) -> TaskKind:
    """Heuristic task label — no LLM call; used for scope gate and logging."""
    lowered = user_input.lower().strip()
    if any(phrase in lowered for phrase in OUT_OF_SCOPE_PHRASES):
        return "out_of_scope"

    has_explore = any(signal in lowered for signal in EXPLORE_SIGNALS)
    has_bugfix = any(signal in lowered for signal in BUGFIX_SIGNALS)
    has_refactor = any(signal in lowered for signal in REFACTOR_SIGNALS)

    if has_explore and not has_bugfix and not has_refactor:
        return "explore"
    if has_refactor:
        return "refactor"
    if has_bugfix:
        return "bugfix"
    return "bugfix"


def is_out_of_scope_request(user_input: str) -> bool:
    """True when the message is unsafe or unbounded for a single agent turn."""
    return classify_task(user_input) == "out_of_scope"


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
