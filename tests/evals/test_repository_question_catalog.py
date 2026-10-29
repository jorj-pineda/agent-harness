import json
import sys

import pytest

from evals.question_grading import canonical_facts
from evals.real_run import acceptance_check, load_tasks, run_task
from harness.config import Settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tools.process import run_process
from workspace import disposable_workspace

TASK_IDS = ["repository_config_precedence", "repository_error_routing"]
ORACLES = {
    "repository_config_precedence": """from service import startup_settings
facts = {
    'defaults': startup_settings({}, {}),
    'environment': startup_settings({'PORT': '9001', 'DEBUG': ' YES '}, {}),
    'zero_flag': startup_settings({'PORT': '9001', 'DEBUG': 'yes'}, {'port': 0, 'debug': False}),
    'none_flag': startup_settings({'PORT': '9001', 'DEBUG': 'yes'}, {'port': None}),
    'string_flag': startup_settings({}, {'debug': 'false'}),
}
""",
    "repository_error_routing": """from pipeline import handle
facts = {
    'transient': handle({'kind': 'transient'}),
    'permanent': handle({'kind': 'permanent'}),
    'empty_string': handle({'kind': 'ok', 'payload': ''}),
    'zero': handle({'kind': 'ok', 'payload': 0}),
    'none': handle({'kind': 'ok', 'payload': None}),
}
try:
    handle({'kind': 'ok'})
except Exception as error:
    facts['missing_payload_exception'] = type(error).__name__
""",
}


@pytest.mark.parametrize("task_id", TASK_IDS)
async def test_question_catalog_reference_facts_match_executable_source(task_id):
    task = next(task for task in load_tasks() if task.id == task_id)
    original = {p.name: p.read_bytes() for p in task.fixture.iterdir() if p.is_file()}
    script = (
        "import sys, json\nsys.path.insert(0, sys.argv[1])\n"
        + ORACLES[task_id]
        + "print(json.dumps(facts))\n"
    )
    with disposable_workspace(task.fixture) as copy:
        result = await run_process(
            [sys.executable, "-I", "-B", "-c", script, str(copy.workspace.root)],
            cwd=copy.workspace.root,
            timeout_seconds=10,
            output_limit=12_000,
        )
    assert result.exit_code == 0, result.stderr
    assert canonical_facts(result.stdout) == canonical_facts(
        (task.reference / "answer.json").read_text()
    )
    assert canonical_facts(result.stdout) == canonical_facts(
        (task.acceptance / "facts.json").read_text()
    )
    assert original == {p.name: p.read_bytes() for p in task.fixture.iterdir() if p.is_file()}


@pytest.mark.parametrize(
    "task_id,key,value",
    [
        ("repository_config_precedence", "zero_flag", {"port": 9001, "debug": True}),
        ("repository_config_precedence", "none_flag", {"port": None, "debug": True}),
        ("repository_config_precedence", "string_flag", {"port": 8000, "debug": False}),
        ("repository_error_routing", "permanent", {"state": "retry", "value": None}),
        ("repository_error_routing", "zero", {"state": "empty", "value": None}),
        ("repository_error_routing", "missing_payload_exception", "RuntimeError"),
    ],
)
async def test_question_catalog_rejects_wrong_precedence_and_failure_facts(task_id, key, value):
    task = next(task for task in load_tasks() if task.id == task_id)
    facts = json.loads((task.reference / "answer.json").read_text())
    facts[key] = value
    with disposable_workspace(task.fixture) as copy:
        result = await acceptance_check(task, copy.workspace.root, answer=json.dumps(facts))
    assert result.exit_code == 1


@pytest.mark.parametrize("task_id", TASK_IDS)
@pytest.mark.parametrize("correct", [False, True])
async def test_catalog_questions_use_inspection_and_score_answer_facts(task_id, correct):
    task = next(task for task in load_tasks() if task.id == task_id)
    facts = json.loads((task.reference / "answer.json").read_text())
    if not correct:
        facts[next(iter(facts))] = "unsupported conclusion"
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(id=f"read-{i}", name="read_file", arguments={"path": path.name})
                for i, path in enumerate(sorted(task.fixture.glob("*.py")))
            ]
        ),
        make_response(content=json.dumps(facts)),
    )
    result = await run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.passed is correct and result.completion_status == "completed"
    assert result.verification_status == "not_run"
    assert result.runtime_config["inspection_only"] is True
    assert result.acceptance_policy == "structured-facts-v1"
    assert not result.modified and not result.added and not result.deleted
