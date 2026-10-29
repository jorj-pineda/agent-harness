from __future__ import annotations

import pytest

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
        ToolCallRecord(name="write_file", arguments={"path": "a.py"}, result={"path": "a.py"}),
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


@pytest.mark.parametrize(
    "prefix", [["pytest"], ["python", "-m", "pytest"], ["python3", "-m", "pytest"]]
)
def test_collection_alias_cannot_certify_checks(prefix):
    argv = [*prefix, "--co", "-q"]
    assert not is_verification_command(argv)
    call = ToolCallRecord(
        name="run_command",
        arguments={"argv": argv},
        result={"success": True, "exit_code": 0},
        latency_ms=1,
    )
    assert verification_status([call]) == "not_run"
    assert not harvest_verification_ran([call])


async def test_real_collection_success_does_not_mean_failing_test_passed(tmp_path):
    import sys

    from tools.process import run_process

    (tmp_path / "test_failing.py").write_text("def test_fails():\n    assert False\n")
    result = await run_process(
        [sys.executable, "-m", "pytest", "--co", "-q"],
        cwd=tmp_path,
        timeout_seconds=30,
        output_limit=12000,
        env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )
    assert result.exit_code == 0
    assert "test_fails" in result.stdout
    call = ToolCallRecord(
        name="run_command",
        arguments={"argv": ["python", "-m", "pytest", "--co", "-q"]},
        result={"exit_code": result.exit_code, "success": True},
    )
    assert verification_status([call]) == "not_run"


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


def test_summary_uses_the_same_unavailable_and_stale_evidence_as_details():
    unavailable = ToolCallRecord(
        name="run_command",
        arguments={"argv": ["pytest", "-q"]},
        error="Tool not executed: token budget exhausted",
    )
    edit = ToolCallRecord(name="write_file", result={"path": "a.py"})
    assert harvest_checks([unavailable])[0].status == "unavailable"
    assert verification_status([unavailable]) == "unavailable"
    assert not harvest_verification_ran([unavailable])
    assert verification_status([unavailable, edit]) == "stale"


def test_invalid_argv_is_not_a_check_in_either_summary_or_details():
    malformed = ToolCallRecord(
        name="run_command", arguments={"argv": ["pytest", 123]}, error="invalid arguments"
    )
    assert harvest_checks([malformed]) == []
    assert verification_status([malformed]) == "not_run"


@pytest.mark.parametrize(
    "result",
    [
        {"success": True, "exit_code": 1},
        {"success": False, "exit_code": 0},
        {"success": True, "exit_code": True},
        {"success": True, "exit_code": "0"},
        {"success": "true", "exit_code": 0},
        {"exit_code": 0},
        {},
    ],
)
def test_inconsistent_check_results_cannot_certify_verification(result):
    before = dict(result)
    call = ToolCallRecord(name="run_command", arguments={"argv": ["pytest", "-q"]}, result=result)
    checks = harvest_checks([call])
    assert checks[0].status == "unavailable" and checks[0].error
    assert verification_status([call]) == "unavailable"
    assert not harvest_verification_ran([call])
    assert call.result == before


@pytest.mark.parametrize(
    "result,status",
    [
        ({"success": True, "exit_code": 0}, "passed"),
        ({"success": False, "exit_code": 1}, "failed"),
        ({"success": False, "exit_code": -15}, "failed"),
        ({"success": True}, "passed"),
        ({"success": False}, "failed"),
    ],
)
def test_consistent_and_legacy_check_results_keep_existing_meaning(result, status):
    call = ToolCallRecord(name="run_command", arguments={"argv": ["pytest", "-q"]}, result=result)
    assert verification_status([call]) == status
    assert harvest_checks([call])[0].error is None
