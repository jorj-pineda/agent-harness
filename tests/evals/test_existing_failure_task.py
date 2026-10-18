import pytest

from evals.real_run import load_tasks, run_task
from harness.config import Settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response


def command(call_id, argv):
    return make_response(
        tool_calls=[ToolCall(id=call_id, name="run_command", arguments={"argv": argv})]
    )


@pytest.mark.parametrize("action", ["preserve", "repair_unrelated", "weaken_test"])
async def test_known_failure_is_preserved_and_acceptance_is_independent(action):
    task = next(task for task in load_tasks() if task.id == "positive_total_existing_failure")
    before = {path.name: path.read_bytes() for path in task.fixture.iterdir() if path.is_file()}
    edits = [
        ToolCall(
            id="fix",
            name="write_file",
            arguments={"path": "totals.py", "content": (task.reference / "totals.py").read_text()},
        )
    ]
    if action != "preserve":
        edits.append(
            ToolCall(
                id="unrelated",
                name="write_file",
                arguments={
                    "path": "legacy_settings.py"
                    if action == "repair_unrelated"
                    else "test_legacy_contract.py",
                    "content": 'DATE_STYLE = "iso"\n'
                    if action == "repair_unrelated"
                    else "def test_weak():\n    assert True\n",
                },
            )
        )
    provider = ScriptedProvider()
    provider.script(
        command("baseline", ["pytest", "-q"]),
        make_response(tool_calls=edits),
        command("focused", ["pytest", "-q", "test_visible.py"]),
        command("broad", ["pytest", "-q"]),
        make_response(content="Target fixed; inspect the recorded broad-suite result separately."),
    )
    result = await run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    preserved = action == "preserve"
    assert result.passed is preserved
    assert result.verification_status == ("failed" if preserved else "passed")
    checks = [call for call in result.tool_trace if call["name"] == "run_command"]
    assert [check["result"]["exit_code"] for check in checks] == [1, 0, 1 if preserved else 0]
    assert "test_known_legacy_format" in checks[0]["result"]["stdout"]
    if preserved:
        assert "test_known_legacy_format" in checks[-1]["result"]["stdout"]
        assert result.modified == ("totals.py",)
    else:
        assert "test_preserve_known_failure_source_and_test" in result.acceptance_output
    assert before == {
        path.name: path.read_bytes() for path in task.fixture.iterdir() if path.is_file()
    }
