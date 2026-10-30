import json

import pytest

from evals import heldout_inputs, heldout_workspaces, real_run
from evals.heldout_guard import HeldoutDriftError
from harness.config import Settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response


async def test_prepared_edit_uses_named_root_preserves_artifacts_and_refuses_reuse(tmp_path):
    task = real_run.load_tasks(heldout_workspaces.REGISTRATIONS[0].root / "tasks.yaml")[0]
    directory = tmp_path / "cell"
    heldout_workspaces.prepare_fixture(task.id, directory)
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={
                        "path": "week_key.py",
                        "content": (task.reference / "week_key.py").read_text(),
                    },
                )
            ]
        ),
        make_response(content="Scripted completion"),
    )
    result = await real_run.run_task(
        task,
        provider=provider,
        settings=Settings(_env_file=None),
        mode="harness",
        prepared_directory=directory,
    )
    assert result.passed and result.modified == ("week_key.py",)
    assert result.runtime_config["prepared_workspace_root"] == str(
        directory.resolve() / "workspace"
    )
    assert result.runtime_config["workspace_policy"] == "prepared-heldout-single-use-v1"
    assert (directory / "admitted.json").is_file()
    assert (directory / "workspace/week_key.py").read_bytes() == (
        task.reference / "week_key.py"
    ).read_bytes()
    calls = len(provider.calls)
    with pytest.raises(HeldoutDriftError):
        await real_run.run_task(
            task,
            provider=provider,
            settings=Settings(_env_file=None),
            mode="harness",
            prepared_directory=directory,
        )
    assert len(provider.calls) == calls


async def test_changed_prepared_root_refuses_before_provider_work(tmp_path):
    task = real_run.load_tasks(heldout_workspaces.REGISTRATIONS[0].root / "tasks.yaml")[0]
    directory = tmp_path / "cell"
    heldout_workspaces.prepare_fixture(task.id, directory)
    (directory / "workspace/week_key.py").write_text("# changed")
    provider = ScriptedProvider()
    with pytest.raises(HeldoutDriftError):
        await real_run.run_task(
            task,
            provider=provider,
            settings=Settings(_env_file=None),
            mode="harness",
            prepared_directory=directory,
        )
    assert provider.calls == [] and not (directory / "admitted.json").exists()


async def test_unregistered_task_cannot_admit_arbitrary_workspace(tmp_path):
    provider = ScriptedProvider()
    with pytest.raises(HeldoutDriftError, match="registered"):
        await real_run.run_task(
            real_run.load_tasks()[0],
            provider=provider,
            settings=Settings(_env_file=None),
            mode="harness",
            prepared_directory=tmp_path,
        )
    assert provider.calls == []


async def test_mock_capture_retains_each_exact_named_root_and_consumes_admission(tmp_path):
    directory = tmp_path / "inputs"
    result = await heldout_inputs.capture_inputs(directory, prepared=True)
    assert len(result["cells"]) == 16 and result["forwarded_requests"] == 0
    for cell in result["cells"]:
        container = directory / f"cell-{cell['ordinal']:02d}"
        root = str(container.resolve() / "workspace")
        assert (container / "admitted.json").is_file()
        assert cell["runtime_config"]["prepared_workspace_root"] == root
        normalized = json.loads((directory / f"{cell['ordinal']:02d}-normalized.json").read_text())
        assert root in normalized["messages"][0]["content"]
        with pytest.raises(HeldoutDriftError):
            heldout_workspaces.verify_prepared(container, task_id=cell["task_id"])
