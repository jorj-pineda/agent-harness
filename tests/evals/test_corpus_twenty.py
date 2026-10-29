import pytest

from evals.real_run import load_tasks, run_task
from harness.config import Settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response

MODULES = {
    "parse_bool_setting": ["bool_settings.py"],
    "iter_batches": ["batching.py"],
    "split_extension": ["filenames.py"],
    "weighted_average": ["stats.py"],
    "extract_line_total": ["pricing.py", "invoice.py"],
}
MUTATIONS = {
    "parse_bool_setting": (".strip().lower()", ".lower()"),
    "iter_batches": ("for item in items:", "for item in list(items):"),
    "split_extension": ("posixpath.splitext(path)", 'posixpath.splitext(path.rsplit("/", 1)[-1])'),
    "weighted_average": ("if weight < 0:", "if False:"),
}


@pytest.mark.parametrize("task_id", MODULES)
@pytest.mark.parametrize("correct", [False, True])
async def test_new_coding_tasks_grade_real_artifacts_beyond_visible_examples(task_id, correct):
    task = next(task for task in load_tasks() if task.id == task_id)
    original = {path.name: path.read_bytes() for path in task.fixture.iterdir() if path.is_file()}
    calls = []
    for index, name in enumerate(MODULES[task_id]):
        content = (task.reference / name).read_text()
        if not correct:
            if task_id == "extract_line_total":
                if name == "invoice.py":
                    content = (task.fixture / name).read_text()  # Unwired second caller.
            else:
                old, new = MUTATIONS[task_id]
                assert old in content
                content = content.replace(old, new)
        calls.append(
            ToolCall(
                id=f"edit-{index}", name="write_file", arguments={"path": name, "content": content}
            )
        )
    provider = ScriptedProvider()
    provider.script(
        make_response(tool_calls=calls),
        make_response(
            tool_calls=[
                ToolCall(id="check", name="run_command", arguments={"argv": ["pytest", "-q"]})
            ]
        ),
        make_response(content="Task implemented; visible check passed."),
    )
    result = await run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.passed is correct and result.completion_status == "completed"
    assert result.verification_status == "passed"
    assert result.acceptance_exit_code == (0 if correct else 1)
    assert result.acceptance_policy == "owned-pytest-v1"
    assert result.category == task.category
    assert result.modified and not result.added and not result.deleted
    assert original == {
        path.name: path.read_bytes() for path in task.fixture.iterdir() if path.is_file()
    }
