import pytest

from evals.real_run import acceptance_check, load_tasks, run_task
from harness.config import Settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from workspace import disposable_workspace


@pytest.mark.parametrize(
    "old,new",
    [
        ('io.StringIO(text, newline="")', "text.splitlines()"),
        ('io.StringIO(text, newline="")', "io.StringIO(text.replace('\\r\\n', '\\n'))"),
        (
            'return list(csv.reader(io.StringIO(text, newline="")))',
            'return [[field.strip() for field in row] for row in csv.reader(io.StringIO(text, newline=""))]',
        ),
        (
            'return list(csv.reader(io.StringIO(text, newline="")))',
            'return [[field for field in row if field] for row in csv.reader(io.StringIO(text, newline=""))]',
        ),
    ],
    ids=[
        "splitlines-loses-embedded-newlines",
        "normalizes-embedded-crlf",
        "strips-spaces",
        "drops-empty-fields",
    ],
)
async def test_csv_criteria_reject_lossy_parsers(old, new):
    task = next(task for task in load_tasks() if task.id == "csv_records")
    original = (task.fixture / "records.py").read_bytes()
    source = (task.reference / "records.py").read_text()
    assert old in source
    with disposable_workspace(task.fixture) as copy:
        (copy.workspace.root / "records.py").write_text(source.replace(old, new))
        result = await acceptance_check(task, copy.workspace.root)
    assert result.exit_code == 1, result.stdout + result.stderr
    assert (task.fixture / "records.py").read_bytes() == original


@pytest.mark.parametrize("solved", [False, True])
async def test_csv_real_tool_patch_is_graded_beyond_visible_example(solved):
    task = next(task for task in load_tasks() if task.id == "csv_records")
    original = {p.name: p.read_bytes() for p in task.fixture.iterdir() if p.is_file()}
    source = (task.reference / "records.py").read_text()
    if not solved:
        source = source.replace('io.StringIO(text, newline="")', "text.splitlines()")
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "records.py", "content": source},
                )
            ]
        ),
        make_response(
            tool_calls=[
                ToolCall(id="check", name="run_command", arguments={"argv": ["pytest", "-q"]})
            ]
        ),
        make_response(content="CSV fix complete; visible check passed."),
    )
    result = await run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.category == "bugfix" and result.passed is solved
    assert result.verification_status == "passed"
    assert result.acceptance_exit_code == (0 if solved else 1)
    assert result.modified == ("records.py",)
    assert original == {p.name: p.read_bytes() for p in task.fixture.iterdir() if p.is_file()}
