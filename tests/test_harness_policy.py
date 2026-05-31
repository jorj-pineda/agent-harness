from __future__ import annotations

from harness.policy import (
    TaskKind,
    classify_task,
    edit_budget_exceeded,
    edit_without_plan,
    is_out_of_scope_request,
)
from harness.state import ToolCallRecord


def test_out_of_scope_detects_destructive_requests() -> None:
    assert is_out_of_scope_request("Please delete .git and start over")
    assert is_out_of_scope_request("run rm -rf / on the server")
    assert is_out_of_scope_request("Delete all tests and rewrite entire repo")


def test_in_scope_coding_request() -> None:
    assert not is_out_of_scope_request("Fix the failing test in test_calc.py")


def test_classify_task_labels_bugfix_explore_refactor() -> None:
    assert classify_task("Fix the failing divide test in test_calc.py") == "bugfix"
    assert classify_task("What arithmetic functions does calc.py define?") == "explore"
    assert classify_task("Refactor inline add docstring in calc.py") == "refactor"


def test_classify_task_marks_unbounded_requests_out_of_scope() -> None:
    assert classify_task("Remove all tests from the project") == "out_of_scope"
    assert classify_task("Rewrite entire repo to use async everywhere") == "out_of_scope"


def test_classify_task_explore_with_fix_signal_is_bugfix() -> None:
    assert classify_task("Find where the bug is and fix the failing test") == "bugfix"


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


def test_task_kind_literal_coverage() -> None:
    kinds: list[TaskKind] = ["bugfix", "explore", "refactor", "out_of_scope"]
    assert len(kinds) == 4
