import pytest

from evals.real_run import acceptance_check, load_tasks, run_task
from harness.config import Settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from workspace import disposable_workspace


@pytest.mark.parametrize(
    "body",
    [
        "return {**overrides, **defaults}",
        "return {**defaults, **{k: v for k, v in overrides.items() if v}}",
        "defaults.update(overrides)\n    return defaults",
        "import copy\n    return copy.deepcopy({**defaults, **overrides})",
    ],
    ids=["wrong-precedence", "truthy-only", "mutates-defaults", "copies-values"],
)
async def test_overlay_criteria_reject_contract_violations(body):
    task = next(task for task in load_tasks() if task.id == "overlay_settings")
    original = (task.fixture / "settings.py").read_bytes()
    with disposable_workspace(task.fixture) as copy:
        (copy.workspace.root / "settings.py").write_text(
            original.decode() + "\n\ndef overlay_settings(defaults, overrides):\n    " + body + "\n"
        )
        result = await acceptance_check(task, copy.workspace.root)
    assert result.exit_code == 1, result.stdout + result.stderr
    assert (task.fixture / "settings.py").read_bytes() == original


@pytest.mark.parametrize("solved", [False, True])
async def test_overlay_real_tool_artifact_is_graded_independently(solved):
    task = next(task for task in load_tasks() if task.id == "overlay_settings")
    original = {p.name: p.read_bytes() for p in task.fixture.iterdir() if p.is_file()}
    content = (task.reference / "settings.py").read_text()
    if not solved:
        content = content.replace(
            "result.update(overrides)", "result.update({k: v for k, v in overrides.items() if v})"
        )
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "settings.py", "content": content},
                )
            ]
        ),
        make_response(
            tool_calls=[
                ToolCall(id="check", name="run_command", arguments={"argv": ["pytest", "-q"]})
            ]
        ),
        make_response(content="Overlay implemented; visible check passed."),
    )
    result = await run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.category == "feature"
    assert result.verification_status == "passed"
    assert result.passed is solved
    assert result.acceptance_exit_code == (0 if solved else 1)
    assert result.modified == ("settings.py",)
    assert original == {p.name: p.read_bytes() for p in task.fixture.iterdir() if p.is_file()}
