from __future__ import annotations

from harness.outcome import (
    harvest_files_touched,
    harvest_patch_summary,
    harvest_verification_ran,
    is_verification_command,
)
from harness.state import ToolCallRecord


def test_is_verification_command_accepts_pytest_ruff_mypy() -> None:
    assert is_verification_command(["pytest", "tests/"])
    assert is_verification_command(["ruff", "check", "."])
    assert is_verification_command(["mypy", "."])
    assert is_verification_command(["python", "-m", "pytest", "test_calc.py"])


def test_is_verification_command_rejects_git_and_shell() -> None:
    assert not is_verification_command(["git", "diff"])
    assert not is_verification_command(["bash", "-c", "pytest"])


def test_harvest_files_touched_collects_successful_writes_in_order() -> None:
    calls = [
        ToolCallRecord(
            name="write_file",
            arguments={"path": "a.py"},
            result={"path": "a.py", "bytes_written": 10},
        ),
        ToolCallRecord(
            name="write_file",
            arguments={"path": "b.py"},
            error="boom",
        ),
        ToolCallRecord(
            name="write_file",
            arguments={"path": "a.py"},
            result={"path": "a.py", "bytes_written": 12},
        ),
        ToolCallRecord(
            name="write_file",
            arguments={"path": "c.py"},
            result={"path": "c.py", "bytes_written": 3},
        ),
    ]

    assert harvest_files_touched(calls) == ["a.py", "c.py"]


def test_harvest_patch_summary_formats_successful_writes() -> None:
    calls = [
        ToolCallRecord(
            name="write_file",
            arguments={"path": "calc.py"},
            result={"path": "calc.py", "bytes_written": 180},
        ),
        ToolCallRecord(
            name="write_file",
            arguments={"path": "other.py"},
            error="denied",
        ),
        ToolCallRecord(
            name="write_file",
            arguments={"path": "note.txt"},
            result={"path": "note.txt", "bytes_written": 12},
        ),
    ]

    assert harvest_patch_summary(calls) == [
        "calc.py (180 bytes written)",
        "note.txt (12 bytes written)",
    ]


def test_harvest_verification_ran_requires_successful_pytest() -> None:
    failing = ToolCallRecord(
        name="run_command",
        arguments={"argv": ["pytest", "test_calc.py"]},
        result={"success": False, "exit_code": 1},
    )
    passing = ToolCallRecord(
        name="run_command",
        arguments={"argv": ["pytest", "test_calc.py"]},
        result={"success": True, "exit_code": 0},
    )

    assert harvest_verification_ran([failing]) is False
    assert harvest_verification_ran([failing, passing]) is True
