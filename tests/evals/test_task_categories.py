from dataclasses import replace

import pytest

from evals.real_run import Task, load_tasks, run_task, summarize, summary_markdown
from harness.config import Settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response


@pytest.mark.parametrize("solved", [False, True])
async def test_feature_acceptance_and_category_survive_visible_test_weakening(solved):
    task = next(task for task in load_tasks() if task.id == "paginate_items")
    original = (task.fixture / "pagination.py").read_bytes()
    reference = (task.reference / "pagination.py").read_text()
    incorrect = (
        "def first_page(values, size):\n    return values[:size]\n\n"
        "def page_items(values, page, page_size):\n"
        "    return values[(page - 1) * page_size:page * page_size]\n"
    )
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="implementation",
                    name="write_file",
                    arguments={
                        "path": "pagination.py",
                        "content": reference if solved else incorrect,
                    },
                ),
                ToolCall(
                    id="weaken",
                    name="write_file",
                    arguments={
                        "path": "test_visible.py",
                        "content": "def test_weak():\n    assert True\n",
                    },
                ),
            ]
        ),
        make_response(
            tool_calls=[
                ToolCall(id="check", name="run_command", arguments={"argv": ["pytest", "-q"]})
            ]
        ),
        make_response(content="Complete"),
    )
    result = await run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.category == "feature" and result.passed is solved
    assert result.verification_status == "passed"  # Editable check is not acceptance.
    assert result.acceptance_exit_code == (0 if solved else 1)
    summary = summarize([result])
    assert summary["harness"]["per_category"] == {
        "feature": {"accepted": int(solved), "attempts": 1}
    }
    assert summary["harness"]["false_completion"] == int(not solved)
    assert f"| feature | {int(solved)}/1 |" in summary_markdown(summary)
    assert (task.fixture / "pagination.py").read_bytes() == original


async def test_existing_bugfix_category_and_mixed_mode_counts_are_independent():
    bugfix = load_tasks()[0]
    assert bugfix.category == "bugfix"
    provider = ScriptedProvider()
    provider.script(make_response(content="No patch"))
    result = await run_task(
        bugfix, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    # Summary-only synthetic rows isolate grouping; this is not extra model evidence.
    feature = replace(result, task_id="paginate_items", category="feature", passed=True)
    other_mode = replace(feature, mode="minimal", passed=False)
    summary = summarize([result, feature, other_mode])
    assert summary["harness"]["per_category"] == {
        "bugfix": {"accepted": 0, "attempts": 1},
        "feature": {"accepted": 1, "attempts": 1},
    }
    assert summary["minimal"]["per_category"] == {"feature": {"accepted": 0, "attempts": 1}}
    assert "| bugfix | 0/1 | 0/0 |" in summary_markdown(summary)


def test_legacy_summary_without_categories_remains_renderable():
    summary = {
        "harness": {
            "per_task": {"old": {"accepted": 1, "attempts": 1}},
            "accepted": 1,
            "attempts": 1,
        }
    }
    assert "| old | 1/1 |" in summary_markdown(summary)
    assert "| Category |" not in summary_markdown(summary)


def test_unsupported_category_is_rejected_before_execution(tmp_path):
    manifest = tmp_path / "tasks.yaml"
    manifest.write_text(
        "- id: divide_zero\n  fixture: divide_zero\n  request: Fix it\n  category: made_up\n"
    )
    with pytest.raises(ValueError):
        load_tasks(manifest)


def test_missing_categories_are_unclassified_in_custom_and_legacy_rows(tmp_path):
    manifest = tmp_path / "tasks.yaml"
    manifest.write_text("- id: divide_zero\n  fixture: divide_zero\n  request: Fix it\n")
    assert load_tasks(manifest)[0].category == "unclassified"
    task = load_tasks()[0]
    assert (
        Task(task.id, task.request, task.fixture, task.acceptance, task.reference).category
        == "unclassified"
    )
