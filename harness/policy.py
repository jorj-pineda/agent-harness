"""Lightweight turn policy: scope gate and edit budget checks."""

from __future__ import annotations

import posixpath
from typing import Literal

from harness.outcome import EDIT_TOOL_NAMES, EMIT_PLAN_TOOL_NAME
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
    """True when a successful file edit ran before any successful emit_plan this turn."""
    saw_plan = False
    for call in tool_calls:
        if call.name == EMIT_PLAN_TOOL_NAME and call.error is None:
            saw_plan = True
        elif call.name in EDIT_TOOL_NAMES and call.error is None and not saw_plan:
            return True
    return False


def edit_precondition_error(
    name: str,
    arguments: dict[str, object],
    tool_calls: list[ToolCallRecord],
    *,
    require_plan: bool,
    max_files: int,
) -> str | None:
    """Reject an edit before dispatch when its turn-level preconditions fail."""
    if name not in EDIT_TOOL_NAMES:
        return None
    if require_plan and not any(
        call.name == EMIT_PLAN_TOOL_NAME and call.error is None for call in tool_calls
    ):
        return "Edit blocked: call emit_plan successfully before editing this turn."
    path = arguments.get("path")
    if max_files < 1 or not isinstance(path, str):
        return None
    attempted = posixpath.normpath(path)
    edited = {
        posixpath.normpath(str(call.result["path"]))
        for call in tool_calls
        if call.name in EDIT_TOOL_NAMES
        and call.error is None
        and isinstance(call.result, dict)
        and isinstance(call.result.get("path"), str)
    }
    if attempted not in edited and len(edited) >= max_files:
        return f"Edit blocked: this turn's {max_files}-file limit is reached."
    return None


def unresolved_edit_blocks(tool_calls: list[ToolCallRecord]) -> list[str]:
    """Paths whose last attempted edit was rejected by a turn-level gate."""
    blocked: set[str] = set()
    for call in tool_calls:
        if call.name not in EDIT_TOOL_NAMES:
            continue
        path = call.arguments.get("path")
        if not isinstance(path, str):
            continue
        normalized = posixpath.normpath(path)
        if call.error is not None and call.error.startswith("Edit blocked:"):
            blocked.add(normalized)
        elif call.error is None:
            blocked.discard(normalized)
    return sorted(blocked)


def repeated_unchanged_call(
    name: str,
    arguments: dict[str, object],
    tool_calls: list[ToolCallRecord],
    *,
    max_identical: int,
) -> bool:
    """Block the next call after identical consecutive requests returned the same outcome."""
    if max_identical < 1 or len(tool_calls) < max_identical:
        return False
    recent = tool_calls[-max_identical:]
    first = recent[0]
    return all(
        call.name == name
        and call.arguments == arguments
        and call.result == first.result
        and call.error == first.error
        for call in recent
    )
