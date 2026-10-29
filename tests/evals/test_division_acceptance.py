import pytest

from evals.real_run import acceptance_check, load_tasks
from workspace import disposable_workspace


@pytest.mark.parametrize(
    "implementation",
    [
        "return a // b",
        "return float(a // b)",
        "return int(a / b)",
        "return a / b if b else 0.0",
    ],
    ids=["floor", "float-floor", "truncate", "swallowed-zero"],
)
async def test_division_acceptance_rejects_wrong_arithmetic(implementation: str) -> None:
    task = next(task for task in load_tasks() if task.id == "divide_zero")
    original = (task.fixture / "calc.py").read_bytes()
    with disposable_workspace(task.fixture) as copy:
        (copy.workspace.root / "calc.py").write_text(f"def divide(a, b):\n    {implementation}\n")
        result = await acceptance_check(task, copy.workspace.root)
    assert result.exit_code == 1, result.stdout + result.stderr
    assert (task.fixture / "calc.py").read_bytes() == original
