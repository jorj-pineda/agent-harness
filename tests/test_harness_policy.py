from __future__ import annotations

from harness.policy import edit_budget_exceeded, is_out_of_scope_request


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
