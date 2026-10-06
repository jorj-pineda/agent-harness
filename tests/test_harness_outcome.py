from __future__ import annotations

from harness.outcome import (
    harvest_checks,
    harvest_files_touched,
    harvest_patch_summary,
    harvest_tool_errors,
    harvest_verification_ran,
    is_verification_command,
    verification_status,
)
from harness.state import ToolCallRecord


def test_check_attempts_distinguish_failure_unavailable_and_stale_evidence() -> None:
    calls = [
        ToolCallRecord(
            name="run_command",
            arguments={"argv": ["pytest", "-q"]},
            result={"exit_code": 1, "success": False},
        ),
        ToolCallRecord(
            name="write_file", arguments={"path": "a.py"}, result={"path": "a.py"}
        ),
        ToolCallRecord(
            name="run_command", arguments={"argv": ["ruff", "check", "."]}, error="missing"
        ),
        ToolCallRecord(
            name="run_command",
            arguments={"argv": ["pytest", "-q"]},
            result={"exit_code": 0, "success": True},
        ),
    ]
    checks = harvest_checks(calls, required_check=["pytest", "-q"])
    assert [(c.status, c.relevant, c.superseded_by_edit, c.exit_code) for c in checks] == [
        ("failed", True, True, 1),
        ("unavailable", False, False, None),
        ("passed", True, False, 0),
    ]
    assert harvest_tool_errors(calls) == ["run_command: missing"]


def test_is_verification_command_accepts_pytest_ruff_mypy() -> None:
    assert is_verification_command(["pytest", "tests/"])
    assert is_verification_command(["ruff", "check", "."])
    assert is_verification_command(["mypy", "."])
    assert is_verification_command(["python", "-m", "pytest", "test_calc.py"])
    assert is_verification_command(["python3", "-m", "pytest", "test_calc.py"])


def test_is_verification_command_rejects_git_and_shell() -> None:
    assert not is_verification_command(["git", "diff"])
    assert not is_verification_command(["bash", "-c", "pytest"])
    assert not is_verification_command(["pytest", "--version"])
    assert not is_verification_command(["ruff", "--help"])
    assert not is_verification_command(["ruff"])
    assert not is_verification_command(["python", "-m", "pytest", "--collect-only"])
    for args in (["--version"], ["--help"], ["--collect-only"]):
        assert not is_verification_command(["python3", "-m", "pytest", *args])
    assert not is_verification_command(["python3", "-m", "unittest"])
    assert not is_verification_command(["python3", "-c", "print(1)"])


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


def test_harvest_targeted_replacement_counts_as_file_edit() -> None:
    calls = [
        ToolCallRecord(
            name="replace_text",
            arguments={"path": "calc.py"},
            result={"path": "calc.py", "replacement_count": 1, "bytes_written": 50},
        )
    ]
    assert harvest_files_touched(calls) == ["calc.py"]
    assert harvest_patch_summary(calls) == ["calc.py (replaced 1 text span)"]


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


def test_verification_tracks_latest_edit_and_later_check_failure() -> None:
    passing = ToolCallRecord(
        name="run_command",
        arguments={"argv": ["pytest", "-q"]},
        result={"success": True},
    )
    failing = ToolCallRecord(
        name="run_command",
        arguments={"argv": ["pytest", "-q"]},
        result={"success": False},
    )
    edit = ToolCallRecord(name="write_file", result={"path": "calc.py"})
    assert verification_status([passing, edit]) == "stale"
    assert not harvest_verification_ran([passing, edit])
    assert verification_status([edit, passing, failing]) == "failed"
    assert not harvest_verification_ran([edit, passing, failing])
    assert verification_status([edit, failing, passing]) == "passed"


def test_configured_check_requires_exact_argv() -> None:
    edit = ToolCallRecord(name="write_file", result={"path": "calc.py"})
    other_check = ToolCallRecord(
        name="run_command",
        arguments={"argv": ["ruff", "check", "."]},
        result={"success": True},
    )
    required_check = ToolCallRecord(
        name="run_command",
        arguments={"argv": ["pytest", "-q"]},
        result={"success": True},
    )
    assert verification_status([edit, other_check], required_check=["pytest", "-q"]) == "not_run"
    assert not harvest_verification_ran([edit, other_check], required_check=["pytest", "-q"])
    assert (
        verification_status([edit, other_check, required_check], required_check=["pytest", "-q"])
        == "passed"
    )


def test_python3_checks_preserve_exact_configuration_and_latest_check_status() -> None:
    argv = ["python3", "-m", "pytest", "-q"]
    passing = ToolCallRecord(
        name="run_command",
        arguments={"argv": argv},
        result={"argv": ["python", *argv[1:]], "success": True, "exit_code": 0},
    )
    failing = ToolCallRecord(
        name="run_command",
        arguments={"argv": argv},
        result={"success": False, "exit_code": 1},
    )
    edit = ToolCallRecord(name="write_file", result={"path": "calc.py"})
    assert verification_status([edit, passing], required_check=argv) == "passed"
    assert verification_status([passing, edit], required_check=argv) == "stale"
    assert verification_status([edit, passing, failing], required_check=argv) == "failed"
    canonical = ["python", *argv[1:]]
    assert verification_status([edit, passing], required_check=canonical) == "not_run"
    checks = harvest_checks([passing], required_check=argv)
    assert [(c.argv, c.status, c.relevant) for c in checks] == [(argv, "passed", True)]
    assert not harvest_checks([passing], required_check=canonical)[0].relevant
