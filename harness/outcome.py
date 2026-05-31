"""Harness-side coding outcome harvesting: edits and verification in tool traces."""

from __future__ import annotations

import logging

from .state import ToolCallRecord

log = logging.getLogger(__name__)

EMIT_PLAN_TOOL_NAME = "emit_plan"
WRITE_FILE_TOOL_NAME = "write_file"
RUN_COMMAND_TOOL_NAME = "run_command"

VERIFICATION_ROOT_COMMANDS = frozenset({"pytest", "ruff", "mypy"})


def is_verification_command(argv: list[str]) -> bool:
    """True when argv is an allowlisted verification invocation."""
    if not argv:
        return False
    root = argv[0]
    if root in VERIFICATION_ROOT_COMMANDS:
        return True
    return root == "python" and len(argv) >= 3 and argv[1] == "-m" and argv[2] == "pytest"


def harvest_files_touched(tool_calls: list[ToolCallRecord]) -> list[str]:
    """Repo-relative paths successfully written this turn, in call order."""
    touched: list[str] = []
    seen: set[str] = set()
    for call in tool_calls:
        if call.name != WRITE_FILE_TOOL_NAME or call.error is not None:
            continue
        result = call.result
        if not isinstance(result, dict):
            continue
        path = result.get("path")
        if isinstance(path, str) and path and path not in seen:
            seen.add(path)
            touched.append(path)
    log.info("outcome_harvest files_touched=%d", len(touched))
    return touched


def harvest_verification_ran(tool_calls: list[ToolCallRecord]) -> bool:
    """True when an allowlisted verification command exited successfully this turn."""
    for call in tool_calls:
        if call.name != RUN_COMMAND_TOOL_NAME or call.error is not None:
            continue
        result = call.result
        if not isinstance(result, dict) or result.get("success") is not True:
            continue
        argv = call.arguments.get("argv")
        if isinstance(argv, list) and is_verification_command([str(a) for a in argv]):
            log.info("outcome_harvest verification_ran=True argv=%s", argv)
            return True
    return False
