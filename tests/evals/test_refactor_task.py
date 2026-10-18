import json

import pytest

from evals.real_run import load_tasks, run_task, write_report
from harness.config import Settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response


@pytest.mark.parametrize("wired", [False, True])
async def test_refactor_acceptance_requires_extraction_and_behavior(tmp_path, wired):
    task = next(task for task in load_tasks() if task.id == "extract_label_normalizer")
    original = (task.fixture / "labels.py").read_text()
    content = (
        (task.reference / "labels.py").read_text()
        if wired
        else original + "\n\ndef normalize_label(value):\n    return value.strip().lower()\n"
    )
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="extract",
                    name="write_file",
                    arguments={"path": "labels.py", "content": content},
                )
            ]
        ),
        make_response(
            tool_calls=[
                ToolCall(id="check", name="run_command", arguments={"argv": ["pytest", "-q"]})
            ]
        ),
        make_response(content="Refactor complete"),
    )
    result = await run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.verification_status == "passed"
    assert result.category == "refactor" and result.passed is wired
    assert result.acceptance_exit_code == (0 if wired else 1)
    if not wired:
        assert "test_both_callers_use_current_module_helper" in result.acceptance_output
    report = tmp_path / "report.json"
    write_report(report, "offline scripted fixture validation", [result])
    evidence = json.loads(report.read_text())
    assert evidence["results"][0]["category"] == "refactor"
    assert evidence["summary"]["harness"]["per_category"] == {
        "refactor": {"accepted": int(wired), "attempts": 1}
    }
    assert (task.fixture / "labels.py").read_text() == original
