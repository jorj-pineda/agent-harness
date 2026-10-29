import hashlib
import json
import shutil
from dataclasses import replace

import pytest
import yaml

from evals import real_run
from evals.real_run import load_tasks, run_task, summarize, summary_markdown
from harness.config import Settings
from tests.api.conftest import ScriptedProvider, make_response


def catalog(tmp_path, partition="held_out"):
    task = load_tasks()[0]
    for kind in ("fixture", "acceptance", "reference"):
        folder = "fixtures" if kind == "fixture" else kind
        shutil.copytree(getattr(task, kind), tmp_path / folder / "sample")
    path = tmp_path / "tasks.yaml"
    document = {
        "format": "task-catalog-v1",
        "id": "test-catalog",
        "partition": partition,
        "tasks": [
            {
                "id": "sample",
                "fixture": "sample",
                "request": task.request,
                "category": task.category,
            }
        ],
    }
    path.write_text(yaml.safe_dump(document))
    return path, document


def test_default_development_and_legacy_custom_identity_are_explicit(tmp_path):
    defaults = load_tasks()
    assert defaults
    assert all(task.task_set.partition == "development" for task in defaults)
    path = tmp_path / "custom.yaml"
    path.write_text("- id: divide_zero\n  fixture: divide_zero\n  request: Fix division\n")
    task = load_tasks(path)[0]
    assert task.fixture == real_run.TASKS_ROOT.resolve() / "fixtures" / "divide_zero"
    assert task.task_set.partition == "unclassified"
    assert task.task_set.manifest_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert replace(task, task_set=None).task_set is None


def test_structured_catalog_uses_its_own_root_and_exact_manifest_bytes(tmp_path):
    path, _ = catalog(tmp_path)
    task = load_tasks(path)[0]
    assert task.id == "sample" and task.fixture == tmp_path / "fixtures" / "sample"
    assert task.task_set.partition == "held_out"
    before = task.task_set.manifest_sha256
    path.write_text(path.read_text() + "\n# metadata-only change\n")
    assert load_tasks(path)[0].task_set.manifest_sha256 != before
    assert task.task_set.manifest_sha256 == before


@pytest.mark.parametrize("defect", ["partition", "extra", "duplicate", "escape", "oversized"])
def test_catalog_refusals_precede_provider_allocation(tmp_path, monkeypatch, defect):
    path, document = catalog(tmp_path)
    if defect == "partition":
        document["partition"] = "untouched"
    elif defect == "extra":
        document["execute"] = True
    elif defect == "duplicate":
        document["tasks"] *= 2
    elif defect == "escape":
        document["tasks"][0]["fixture"] = "../sample"
    path.write_text(yaml.safe_dump(document) if defect != "oversized" else "x" * 1_000_001)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid catalog allocated a provider")

    monkeypatch.setattr(real_run, "build_configured_provider", forbidden)
    with pytest.raises(ValueError):
        real_run.main(["--task-manifest", str(path)])


def test_catalog_reference_directory_cannot_escape_through_symlink(tmp_path):
    path, _ = catalog(tmp_path)
    root = tmp_path / "reference"
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    root.rename(outside)
    root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        load_tasks(path)


def test_validate_only_selects_one_explicit_catalog_without_provider(tmp_path, monkeypatch):
    path, _ = catalog(tmp_path)
    monkeypatch.setattr(
        real_run,
        "build_configured_provider",
        lambda *args, **kwargs: pytest.fail("validation allocated provider"),
    )
    assert real_run.main(["--task-manifest", str(path), "--validate-only"]) == 0


async def test_catalog_identity_survives_real_task_report_and_partition_summary(tmp_path):
    path, _ = catalog(tmp_path)
    task = load_tasks(path)[0]
    provider = ScriptedProvider()
    provider.script(make_response(content="Not fixed"))
    result = await run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.task_set == task.task_set.model_dump()
    legacy = replace(result, task_id="unknown-project", task_set=None)
    development = replace(
        result, task_id="development-task", task_set=load_tasks()[0].task_set.model_dump()
    )
    summary = summarize([result, legacy, development])
    assert set(summary["harness"]["per_partition"]) == {"development", "held_out", "unclassified"}
    assert "| held_out | 0/1 |" in summary_markdown(summary)
    report = tmp_path / "report.json"
    real_run.write_report(report, "scripted catalog contract", [result])
    assert json.loads(report.read_text())["results"][0]["task_set"] == task.task_set.model_dump()
