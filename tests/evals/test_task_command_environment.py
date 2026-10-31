import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from evals import real_run
from evals.heldout_comparison import validate_plan
from evals.heldout_rehearsal import CannedProvider
from evals.heldout_settings import protocol_settings


def environment():
    return {
        "PATH": os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"]),
        "LC_ALL": "C",
    }


async def test_controls_tools_and_grader_use_explicit_environment(tmp_path, monkeypatch):
    import tools.code as code
    from evals.heldout_guard import REGISTRATIONS

    task = real_run.load_tasks(REGISTRATIONS[0].root / "tasks.yaml")[0]
    plan = validate_plan()
    selected = environment()
    observed = []
    run = real_run.run_process

    async def checked(argv, **kwargs):
        assert kwargs["inherit_environment"] is False
        assert kwargs["env"] == real_run._task_command_environment(selected)
        observed.append(list(argv))
        return await run(argv, **kwargs)

    monkeypatch.setattr(real_run, "run_process", checked)
    monkeypatch.setattr(code, "run_process", checked)
    monkeypatch.setenv("LANG", "ambient-sentinel")
    await real_run.validate_task(task, command_env=selected)
    result = await real_run.run_task(
        task,
        provider=CannedProvider(task, plan),
        settings=protocol_settings(plan),
        mode="harness",
        command_env=selected,
    )
    assert result.passed and result.completion_status == "completed"
    assert len(observed) == 6  # Four controls, one tool check, one independent grade.
    expected = hashlib.sha256(
        json.dumps(real_run._task_command_environment(selected), sort_keys=True).encode()
    ).hexdigest()
    assert result.runtime_config["command_environment"] == {
        "policy": "replacement-v1",
        "sha256": expected,
    }
    assert "ambient-sentinel" not in json.dumps(result.runtime_config)


@pytest.mark.parametrize(
    "selected",
    [
        {},
        {"PATH": "relative"},
        {"PATH": "/bin", "API_KEY": "private"},
        {"PATH": "/bin", "LANG": "bad\x00value"},
    ],
)
async def test_invalid_environment_refuses_before_workspace_admission(selected, monkeypatch):
    from evals.heldout_guard import REGISTRATIONS

    task = real_run.load_tasks(REGISTRATIONS[0].root / "tasks.yaml")[0]
    monkeypatch.setattr(
        real_run, "_heldout_guard", lambda *a, **k: pytest.fail("workspace preparation")
    )
    with pytest.raises(ValueError):
        await real_run.run_task(
            task,
            provider=None,
            settings=protocol_settings(validate_plan()),
            mode="harness",
            command_env=selected,
        )


def test_mapping_is_copied_and_required_pytest_flags_win():
    selected = environment()
    selected["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "0"
    result = real_run._task_command_environment(selected)
    selected["PATH"] = "later-change"
    assert result["PATH"] != selected["PATH"]
    assert result["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert result["PYTHONDONTWRITEBYTECODE"] == "1"
