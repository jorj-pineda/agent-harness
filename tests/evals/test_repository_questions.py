import json
import sys
from dataclasses import replace

import pytest

from evals.question_grading import MAX_FACT_BYTES, canonical_facts
from evals.real_run import acceptance_check, load_tasks, run_task, summarize, summary_markdown
from harness.config import Settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tools.process import run_process
from workspace import disposable_workspace


def question():
    return next(task for task in load_tasks() if task.id == "repository_retry_policy")


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "{}",
        "[]",
        "null",
        '{"n": 1, "n": 1}',
        '{"n": NaN}',
        '```json\n{"n": 1}\n```',
        ' {"n": "' + "x" * MAX_FACT_BYTES + '"}',
    ],
)
def test_invalid_structured_fact_envelopes_refuse(answer):
    with pytest.raises(ValueError):
        canonical_facts(answer)


def test_fact_byte_bound_counts_utf8_and_keeps_exact_limit():
    text = '{"n":"é"}'
    text += " " * (MAX_FACT_BYTES - len(text.encode("utf-8")))
    assert canonical_facts(text) == '{"n": "\\u00e9"}'
    with pytest.raises(ValueError, match="byte limit"):
        canonical_facts(text + " ")


@pytest.mark.parametrize(
    "defect",
    [
        "stale-docs",
        "wrong-boundary",
        "boolean-for-int",
        "float-for-int",
        "extra-key",
        "numeric-string",
    ],
)
async def test_question_rejects_incorrect_or_mistyped_facts(defect):
    task = question()
    facts = json.loads((task.reference / "answer.json").read_text())
    if defect == "stale-docs":
        facts.update(default_max_attempts=5, default_delays=[3, 3, 3, 3])
    elif defect == "wrong-boundary":
        facts["retry_after_attempts"] = [True, True, True]
    elif defect == "boolean-for-int":
        facts["zero_max_attempts"] = True
    elif defect == "float-for-int":
        facts["zero_max_attempts"] = 1.0
    elif defect == "extra-key":
        facts["unrequested"] = 1
    else:
        facts["default_max_attempts"] = "3"
    with disposable_workspace(task.fixture) as copy:
        result = await acceptance_check(task, copy.workspace.root, answer=json.dumps(facts))
    assert result.exit_code == 1


@pytest.mark.parametrize("change", ["content", "mode", "added", "deleted"])
async def test_correct_facts_cannot_pass_with_changed_scoped_workspace(change):
    task = question()
    with disposable_workspace(task.fixture) as copy:
        path = copy.workspace.root / "worker.py"
        if change == "content":
            path.write_text("# altered\n")
        elif change == "mode":
            path.chmod(0o600)
        elif change == "added":
            (copy.workspace.root / "extra.py").write_text("# added\n")
        else:
            path.unlink()
        result = await acceptance_check(
            task, copy.workspace.root, answer=(task.reference / "answer.json").read_text()
        )
    assert result.exit_code == 1 and "workspace: changed" in result.stdout


async def test_reference_facts_match_executable_fixture_without_modifying_source():
    task = question()
    original = {p.name: p.read_bytes() for p in task.fixture.iterdir() if p.is_file()}
    # Trusted fixture oracle in a separate isolated interpreter/disposable copy,
    # not model-invoked execution or a judge.
    script = """import json, sys
sys.path.insert(0, sys.argv[1])
from retry_policy import max_attempts, delay_seconds
from worker import should_retry
print(json.dumps({
    'default_max_attempts': max_attempts({}),
    'default_delays': [delay_seconds(i, {}) for i in range(1, 5)],
    'retry_after_attempts': [should_retry(i, {}) for i in range(1, 4)],
    'zero_max_attempts': max_attempts({'max_attempts': 0}),
    'zero_base_delays': [delay_seconds(i, {'base_delay': 0}) for i in (1, 2)],
}))
"""
    with disposable_workspace(task.fixture) as copy:
        result = await run_process(
            [sys.executable, "-I", "-B", "-c", script, str(copy.workspace.root)],
            cwd=copy.workspace.root,
            timeout_seconds=10,
            output_limit=12_000,
        )
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == json.loads((task.reference / "answer.json").read_text())
    assert json.loads(result.stdout) == json.loads((task.acceptance / "facts.json").read_text())
    assert original == {p.name: p.read_bytes() for p in task.fixture.iterdir() if p.is_file()}


async def test_inspection_question_uses_shared_tools_and_separate_answer_acceptance():
    task = question()
    # Reorder keys/whitespace: semantic JSON facts, not prose/token similarity.
    facts = json.loads((task.reference / "answer.json").read_text())
    answer = json.dumps(dict(reversed(list(facts.items()))), indent=2)
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(id=f"read-{i}", name="read_file", arguments={"path": name})
                for i, name in enumerate(("retry_config.py", "retry_policy.py", "worker.py"))
            ]
        ),
        make_response(content=answer),
    )
    result = await run_task(
        task,
        provider=provider,
        settings=Settings(_env_file=None, require_verification_before_finish=True),
        mode="harness",
    )
    assert result.passed and result.completion_status == "completed"
    assert result.verification_status == "not_run"
    assert result.acceptance_policy == "structured-facts-v1"
    assert result.runtime_config["inspection_only"] is True
    assert result.acceptance_identity["status"] == "matching"
    assert not result.added and not result.modified and not result.deleted
    names = {spec["name"] for spec in result.runtime_config["tool_specs"]}
    assert "read_file" in names and "write_file" not in names and "run_command" not in names
    artifact = replace(result, task_id="synthetic-coding-row", category="bugfix", passed=False)
    summary = summarize([result, artifact])
    assert summary["harness"]["per_acceptance_subject"] == {
        "artifact": {"accepted": 0, "completed_and_accepted": 0, "attempts": 1},
        "answer": {"accepted": 1, "completed_and_accepted": 1, "attempts": 1},
    }
    rendered = summary_markdown(summary)
    assert "| artifact | 0/1 |" in rendered and "| answer | 1/1 |" in rendered
    assert "Artifact acceptance (independent final-state checks)" not in rendered
    summary["minimal"] = {
        "per_task": {"legacy": {"accepted": 0, "attempts": 1}},
        "accepted": 0,
        "attempts": 1,
    }
    assert "| answer | 1/1 | unknown |" in summary_markdown(summary)


async def test_question_attempted_write_is_refused_without_mutation():
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="write",
                    name="write_file",
                    arguments={"path": "worker.py", "content": "changed"},
                )
            ]
        )
    )
    result = await run_task(
        question(), provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert not result.passed and result.completion_status != "completed"
    assert not result.modified and not result.added and not result.deleted
    assert result.completion_status == "blocked" and result.tool_trace == []


async def test_workspace_owned_fact_rubric_is_refused():
    task = question()
    with disposable_workspace(task.fixture) as copy:
        rubric = copy.workspace.root / "rubric"
        rubric.mkdir()
        (rubric / "facts.json").write_text('{"n": 1}')
        with pytest.raises(ValueError, match="outside"):
            await acceptance_check(
                replace(task, acceptance=rubric), copy.workspace.root, answer='{"n":1}'
            )
