import json
import sys
from pathlib import Path

import pytest

from evals.question_grading import canonical_facts
from evals.real_run import load_tasks, run_task, validate_task
from harness.config import Settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tools.process import run_process
from workspace import disposable_workspace

MANIFEST = Path(__file__).parents[2] / "evals/heldout_tasks/v1/tasks.yaml"
MODULES = {
    "ho_iso_week": "week_key.py",
    "ho_rolling_mean": "rolling.py",
    "ho_endpoint_record": "endpoint.py",
}
MUTATIONS = {
    "ho_iso_week": (
        "year, week, _ = day.isocalendar()",
        "year, week, _ = day.isocalendar()\n    if day.month != 1:\n        year = day.year",
    ),
    "ho_rolling_mean": ("return result", "return result[:1]"),
    "ho_endpoint_record": ("Endpoint(host, int(port))", "Endpoint(host.strip(), int(port))"),
}


@pytest.mark.parametrize("task_id", [task.id for task in load_tasks(MANIFEST)])
async def test_heldout_initial_controls_fail_and_references_pass(task_id):
    await validate_task(next(task for task in load_tasks(MANIFEST) if task.id == task_id))


def test_heldout_is_explicit_and_separate_from_development():
    heldout = load_tasks(MANIFEST)
    assert len(heldout) == 4
    assert {task.category for task in heldout} == {
        "bugfix",
        "feature",
        "refactor",
        "repository_question",
    }
    assert all(task.task_set.partition == "held_out" for task in heldout)
    assert not {task.id for task in heldout} & {task.id for task in load_tasks()}
    assert all(task.task_set.partition == "development" for task in load_tasks())


@pytest.mark.parametrize("task_id", MODULES)
@pytest.mark.parametrize("correct", [False, True])
async def test_heldout_coding_artifact_controls_use_shared_runtime(task_id, correct):
    task = next(task for task in load_tasks(MANIFEST) if task.id == task_id)
    before = {path.name: path.read_bytes() for path in task.fixture.iterdir() if path.is_file()}
    name = MODULES[task_id]
    content = (task.reference / name).read_text()
    if not correct:
        old, new = MUTATIONS[task_id]
        assert old in content
        content = content.replace(old, new)
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(id="edit", name="write_file", arguments={"path": name, "content": content})
            ]
        ),
        make_response(
            tool_calls=[
                ToolCall(id="check", name="run_command", arguments={"argv": ["pytest", "-q"]})
            ]
        ),
        make_response(content="Task complete; visible check passed."),
    )
    result = await run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.passed is correct and result.completion_status == "completed"
    assert result.verification_status == "passed"
    assert result.task_set["partition"] == "held_out"
    assert before == {
        path.name: path.read_bytes() for path in task.fixture.iterdir() if path.is_file()
    }


async def test_heldout_cache_reference_facts_match_executable_source():
    task = next(task for task in load_tasks(MANIFEST) if task.id == "ho_cache_policy")
    script = """import sys, json
sys.path.insert(0, sys.argv[1])
from cache import Cache
from client import read_value
cache = Cache()
keys = []
def fetch(key):
    keys.append(key)
    return False
values = [read_value(cache, 'key', fetch, now=now) for now in (10, 14, 15)]
boundary = [cache.get('key', now=20), cache.get('key', now=20)]
cache.put('zero', None, now=30, ttl=0)
zero = cache.get('zero', now=30)
cache.put('none', None, now=40, ttl=5)
print(json.dumps(dict(resolved_values=values, fetch_keys=keys, boundary_reads=boundary,
    zero_ttl_read=zero, none_hit=cache.get('none', now=41))))
"""
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


@pytest.mark.parametrize("correct", [False, True])
async def test_heldout_question_controls_are_inspection_only(correct):
    task = next(task for task in load_tasks(MANIFEST) if task.id == "ho_cache_policy")
    facts = json.loads((task.reference / "answer.json").read_text())
    if not correct:
        facts["fetch_keys"] = ["key", "key", "key"]  # Incorrect false-value cache miss.
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[ToolCall(id="read", name="read_file", arguments={"path": "cache.py"})]
        ),
        make_response(content=json.dumps(facts)),
    )
    result = await run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.passed is correct and result.completion_status == "completed"
    assert result.verification_status == "not_run"
    assert result.runtime_config["inspection_only"] is True
    assert not result.modified and not result.added and not result.deleted
