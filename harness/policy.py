"""Lightweight turn policy: scope gate and edit budget checks."""

from __future__ import annotations

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
