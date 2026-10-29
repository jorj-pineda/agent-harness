import pytest

from evals.real_run import acceptance_check, load_tasks, run_task
from harness.config import Settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from workspace import disposable_workspace


@pytest.mark.parametrize(
    "old,new",
    [
        ("time.time() if clock is None else clock()", "time.time()"),
        ("clock=None", "clock=time.time"),
        ("now >= deadline", "now > deadline"),
        ("else clock()", "else (clock(), clock())[1]"),
    ],
    ids=["ignores-injection", "captures-default", "strict-expiry", "double-clock-read"],
)
async def test_clock_criteria_reject_contract_violations(old, new):
    task = next(task for task in load_tasks() if task.id == "inject_clock")
    original = (task.fixture / "deadlines.py").read_bytes()
    source = (task.reference / "deadlines.py").read_text()
    assert old in source
    with disposable_workspace(task.fixture) as copy:
        (copy.workspace.root / "deadlines.py").write_text(source.replace(old, new))
        result = await acceptance_check(task, copy.workspace.root)
    assert result.exit_code == 1, result.stdout + result.stderr
    assert (task.fixture / "deadlines.py").read_bytes() == original


@pytest.mark.parametrize("solved", [False, True])
async def test_clock_refactor_real_tool_artifact_has_independent_boundary_checks(solved):
    task = next(task for task in load_tasks() if task.id == "inject_clock")
    original = {p.name: p.read_bytes() for p in task.fixture.iterdir() if p.is_file()}
    source = (task.reference / "deadlines.py").read_text()
    if not solved:
        source = source.replace("now >= deadline", "now > deadline")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "deadlines.py", "content": source},
                )
            ]
        ),
        make_response(
            tool_calls=[
                ToolCall(id="check", name="run_command", arguments={"argv": ["pytest", "-q"]})
            ]
        ),
        make_response(content="Clock refactor complete; visible check passed."),
    )
    result = await run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.category == "refactor" and result.passed is solved
    assert result.verification_status == "passed"
    assert result.acceptance_exit_code == (0 if solved else 1)
    assert result.modified == ("deadlines.py",)
    assert original == {p.name: p.read_bytes() for p in task.fixture.iterdir() if p.is_file()}
