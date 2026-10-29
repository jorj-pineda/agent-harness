"""Synthetic aggregation controls, not model-performance observations."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from evals.real_run import load_tasks, run_task, summarize, summary_markdown, write_report
from harness.config import Settings
from tests.api.conftest import ScriptedProvider, make_response


@pytest.fixture
async def row():
    provider = ScriptedProvider()
    provider.script(make_response(content="No patch", prompt_tokens=10, completion_tokens=5))
    return await run_task(
        load_tasks()[0], provider=provider, settings=Settings(_env_file=None), mode="harness"
    )


async def test_models_and_providers_keep_separate_outcomes_categories_partitions_and_usage(
    row, tmp_path
):
    rows = [
        replace(row, provider="local", model="same", passed=True, termination="accepted"),
        replace(
            row,
            provider="local",
            model="same",
            completion_status="budget_exhausted",
            termination="budget_exhausted",
            prompt_tokens=None,
            completion_tokens=None,
        ),
        replace(
            row,
            provider="local",
            model="other",
            category="repository_question",
            passed=True,
            completion_status="incomplete",
            termination="iteration_limit",
            task_set={**row.task_set, "partition": "held_out"},
        ),
        replace(row, provider="hosted", model="same", category="feature", task_set=None),
        replace(row, provider="local", model="unknown", mode="minimal", task_set=None),
    ]
    report = tmp_path / "report.json"
    write_report(report, "synthetic aggregation controls", rows)
    data = json.loads(report.read_text())
    summary = data["summary"]
    models = summary["harness"]["per_model"]
    assert set(models) == {"local", "hosted"}
    assert set(models["local"]) == {"same", "other"}
    local = models["local"]["same"]
    assert (local["attempts"], local["accepted"], local["completed_and_accepted"]) == (2, 1, 1)
    assert local["per_category"]["bugfix"]["attempts"] == 2
    assert local["per_partition"]["development"]["attempts"] == 2
    assert local["terminations"] == {"accepted": 1, "budget_exhausted": 1}
    assert local["median_total_tokens"] == 15
    assert local["unreported_token_attempts"] == 1
    question = models["local"]["other"]
    assert question["per_acceptance_subject"]["answer"]["accepted"] == 1
    assert question["per_acceptance_subject"]["artifact"]["attempts"] == 0
    assert question["completed_and_accepted"] == 0
    assert question["per_partition"]["held_out"]["attempts"] == 1
    assert models["hosted"]["same"]["per_category"]["feature"]["accepted"] == 0
    assert models["hosted"]["same"]["per_partition"]["unclassified"]["attempts"] == 1
    assert set(summary["minimal"]["per_model"]["local"]) == {"unknown"}
    assert summary["harness"]["attempts"] == 4 and summary["harness"]["accepted"] == 2
    assert sum(group["attempts"] for groups in models.values() for group in groups.values()) == 4
    assert [(r["provider"], r["model"]) for r in data["results"]] == [
        (r.provider, r.model) for r in rows
    ]
    rendered = summary_markdown(summary)
    assert "| local | same | harness | bugfix | 1/2 | 1/2 |" in rendered
    assert "| local | other | harness | repository_question | 1/1 | 0/1 |" in rendered
    assert "runtime settings and budgets" in rendered


async def test_model_keys_are_not_lossily_concatenated_and_table_labels_are_escaped(row):
    summary = summarize(
        [
            replace(row, provider="a/b", model="c"),
            replace(row, provider="a", model="b/c"),
            replace(row, provider="a", model="pipe|model\nline"),
        ]
    )
    models = summary["harness"]["per_model"]
    assert models["a/b"]["c"]["attempts"] == 1
    assert models["a"]["b/c"]["attempts"] == 1
    assert "| a | pipe\\|model line | harness |" in summary_markdown(summary)


async def test_legacy_breakdowns_remain_unknown_without_mutating_summary(row):
    summary = summarize([row, replace(row, mode="minimal")])
    del summary["minimal"]["per_model"]
    before = deepcopy(summary)
    assert "| unknown | unknown | minimal | unknown | unknown | unknown |" in summary_markdown(
        summary
    )
    assert summary == before
    del summary["harness"]["per_model"]
    assert "Provider/model breakdown: unknown" in summary_markdown(summary)


def test_empty_results_have_no_invented_model_groups():
    assert summarize([]) == {}
