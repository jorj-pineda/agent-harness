"""Harness-side coding outcome harvesting: edits and verification in tool traces."""

from __future__ import annotations

import logging

from .outcome_types import VerificationStatus
from .state import CheckRecord, ToolCallRecord

log = logging.getLogger(__name__)

EMIT_PLAN_TOOL_NAME = "emit_plan"
WRITE_FILE_TOOL_NAME = "write_file"
REPLACE_TEXT_TOOL_NAME = "replace_text"
EDIT_TOOL_NAMES = frozenset({WRITE_FILE_TOOL_NAME, REPLACE_TEXT_TOOL_NAME})
RUN_COMMAND_TOOL_NAME = "run_command"

VERIFICATION_ROOT_COMMANDS = frozenset({"pytest", "ruff", "mypy"})


def is_verification_command(argv: list[str]) -> bool:
    """True when argv is an allowlisted verification invocation."""
    if not argv:
        return False
    if any(arg in {"--version", "-V", "--help", "-h", "--collect-only"} for arg in argv[1:]):
        return False
    root = argv[0]
    if root == "ruff":
        return len(argv) >= 2 and argv[1] == "check"
    if root in VERIFICATION_ROOT_COMMANDS:
        return True
    return root == "python" and len(argv) >= 3 and argv[1] == "-m" and argv[2] == "pytest"


def harvest_files_touched(tool_calls: list[ToolCallRecord]) -> list[str]:
    """Repo-relative paths successfully written this turn, in call order."""
    touched: list[str] = []
    seen: set[str] = set()
    for call in tool_calls:
        if call.name not in EDIT_TOOL_NAMES or call.error is not None:
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


def harvest_patch_summary(tool_calls: list[ToolCallRecord]) -> list[str]:
    """One-line summaries for each successful file edit this turn."""
    summaries: list[str] = []
    for call in tool_calls:
        if call.name not in EDIT_TOOL_NAMES or call.error is not None:
            continue
        result = call.result
        if not isinstance(result, dict):
            continue
        path = result.get("path")
        bytes_written = result.get("bytes_written")
        if not isinstance(path, str) or not path:
            continue
        if call.name == REPLACE_TEXT_TOOL_NAME:
            summaries.append(f"{path} (replaced 1 text span)")
        elif isinstance(bytes_written, int):
            summaries.append(f"{path} ({bytes_written} bytes written)")
        else:
            summaries.append(path)
    log.info("outcome_harvest patch_summary=%d", len(summaries))
    return summaries


def harvest_checks(
    tool_calls: list[ToolCallRecord], *, required_check: list[str] | None = None
) -> list[CheckRecord]:
    """List observed verification attempts; a later edit supersedes earlier evidence."""
    checks: list[CheckRecord] = []
    for call in tool_calls:
        if call.name in EDIT_TOOL_NAMES and call.error is None and isinstance(call.result, dict):
            for check in checks:
                check.superseded_by_edit = True
            continue
        argv = call.arguments.get("argv")
        if call.name != RUN_COMMAND_TOOL_NAME or not isinstance(argv, list):
            continue
        if not all(isinstance(token, str) for token in argv) or not is_verification_command(argv):
            continue
        result = call.result if isinstance(call.result, dict) else {}
        exit_code = result.get("exit_code")
        checks.append(
            CheckRecord(
                argv=argv,
                exit_code=exit_code if isinstance(exit_code, int) else None,
                status=(
                    "unavailable"
                    if call.error is not None
                    else "passed"
                    if result.get("success") is True
                    else "failed"
                ),
                relevant=required_check is None or argv == required_check,
                error=call.error,
            )
        )
    return checks


def harvest_tool_errors(tool_calls: list[ToolCallRecord]) -> list[str]:
    """Observed tool errors in call order; later actions may have recovered from them."""
    return [f"{call.name}: {call.error}" for call in tool_calls if call.error is not None]


def harvest_verification_ran(
    tool_calls: list[ToolCallRecord], *, required_check: list[str] | None = None
) -> bool:
    """True when the latest relevant check passed after the latest file edit."""
    return verification_status(tool_calls, required_check=required_check) == "passed"


def verification_status(
    tool_calls: list[ToolCallRecord], *, required_check: list[str] | None = None
) -> VerificationStatus:
    """Summarize the latest check, invalidating it when a later edit succeeds."""
    status: VerificationStatus = "not_run"
    for call in tool_calls:
        if call.name in EDIT_TOOL_NAMES and call.error is None and isinstance(call.result, dict):
            status = "stale" if status != "not_run" else "not_run"
            continue
        if call.name != RUN_COMMAND_TOOL_NAME:
            continue
        argv = call.arguments.get("argv")
        if not isinstance(argv, list) or not is_verification_command([str(a) for a in argv]):
            continue
        if required_check is not None and argv != required_check:
            continue
        result = call.result
        status = (
            "passed"
            if call.error is None and isinstance(result, dict) and result.get("success") is True
            else "failed"
        )
    return status
