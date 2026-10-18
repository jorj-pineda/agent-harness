import pytest

from evals.real_run import load_tasks, run_task
from harness.config import Settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response


@pytest.mark.parametrize("solution", ["both", "parser_only", "client_only"])
async def test_both_public_interfaces_must_work_in_final_artifact(solution):
    task = next(task for task in load_tasks() if task.id == "retry_after_seconds")
    before = {path.name: path.read_bytes() for path in task.fixture.iterdir() if path.is_file()}
    files = (
        ["retry_after.py", "retry_client.py"]
        if solution == "both"
        else ["retry_after.py" if solution == "parser_only" else "retry_client.py"]
    )
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id=f"edit-{index}",
                    name="write_file",
                    arguments={"path": name, "content": (task.reference / name).read_text()},
                )
                for index, name in enumerate(files)
            ]
        ),
        make_response(content="Feature complete"),
    )
    result = await run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.passed is (solution == "both")
    assert result.category == "feature" and result.acceptance_policy == "owned-pytest-v1"
    assert result.acceptance_exit_code == (0 if solution == "both" else 1)
    assert before == {
        path.name: path.read_bytes() for path in task.fixture.iterdir() if path.is_file()
    }
