from __future__ import annotations

from harness.policy import edit_budget_exceeded, edit_without_plan, is_out_of_scope_request
from harness.state import ToolCallRecord


def test_out_of_scope_detects_destructive_requests() -> None:
    assert is_out_of_scope_request("Please delete .git and start over")
    assert is_out_of_scope_request("run rm -rf / on the server")


def test_in_scope_coding_request() -> None:
    assert not is_out_of_scope_request("Fix the failing test in test_calc.py")


def test_edit_budget_exceeded_when_too_many_files() -> None:
    files = ["a.py", "b.py", "c.py"]
    assert edit_budget_exceeded(files, max_files=2)
    assert not edit_budget_exceeded(files, max_files=3)
    assert not edit_budget_exceeded(files, max_files=0)


def test_edit_without_plan_when_write_precedes_emit_plan() -> None:
    calls = [
        ToolCallRecord(
            name="write_file",
            arguments={"path": "calc.py"},
            result={"path": "calc.py", "bytes_written": 10},
        ),
    ]
    assert edit_without_plan(calls)


def test_edit_without_plan_false_after_successful_emit_plan() -> None:
    calls = [
        ToolCallRecord(
            name="emit_plan",
            arguments={"steps": ["read", "fix", "verify"]},
            result={"steps": ["read", "fix", "verify"], "step_count": 3},
        ),
        ToolCallRecord(
            name="write_file",
            arguments={"path": "calc.py"},
            result={"path": "calc.py", "bytes_written": 10},
        ),
    ]
    assert not edit_without_plan(calls)
