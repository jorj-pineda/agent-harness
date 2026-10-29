import json
import shutil
from dataclasses import asdict, replace

import pytest

from evals import real_run
from evals.real_run import RealTaskResult, load_tasks, run_task, write_report
from harness.config import Settings
from tests.api.conftest import ScriptedProvider, make_response


async def observed_run(task):
    provider = ScriptedProvider()
    provider.script(make_response(content="No patch"))
    return await run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )


async def test_report_distinguishes_criteria_with_the_same_starting_fixture(tmp_path):
    original = load_tasks()[0]
    criteria = tmp_path / "acceptance"
    shutil.copytree(original.acceptance, criteria)
    task = replace(original, acceptance=criteria)
    first = await observed_run(task)
    path = criteria / "test_acceptance.py"
    path.write_bytes(path.read_bytes() + b"\n# Criterion revision\n")
    second = await observed_run(task)
    assert first.source_revision == second.source_revision
    assert first.acceptance_identity["status"] == second.acceptance_identity["status"] == "matching"
    assert first.acceptance_identity["before_sha256"] != second.acceptance_identity["before_sha256"]
    assert first.passed is second.passed is False
    report = tmp_path / "report.json"
    write_report(report, "scripted criterion identity", [first, second])
    rows = json.loads(report.read_text())["results"]
    assert rows[0]["acceptance_identity"] == first.acceptance_identity
    assert rows[1]["acceptance_identity"] == second.acceptance_identity
    legacy = asdict(first)
    legacy.pop("acceptance_identity")
    assert RealTaskResult(**legacy).acceptance_identity is None


@pytest.mark.parametrize("passing", [False, True])
async def test_criteria_drift_during_grading_remains_explicit(monkeypatch, tmp_path, passing):
    original = load_tasks()[0]
    criteria = tmp_path / "acceptance"
    shutil.copytree(original.acceptance, criteria)
    grader = real_run.acceptance_check

    async def changed_after_grade(task, root):
        result = await grader(task, root)
        (criteria / "extra.py").write_text("# Changed during grading\n")
        return result

    monkeypatch.setattr(real_run, "acceptance_check", changed_after_grade)
    fixture = tmp_path / "fixture"
    shutil.copytree(original.fixture, fixture)
    if passing:
        shutil.copyfile(original.reference / "calc.py", fixture / "calc.py")
    result = await observed_run(replace(original, fixture=fixture, acceptance=criteria))
    assert result.acceptance_identity["status"] == "changed"
    assert result.acceptance_identity["before_sha256"] != result.acceptance_identity["after_sha256"]
    assert result.passed is False
    assert result.acceptance_exit_code == (0 if passing else 1)
    assert result.termination == "acceptance_invalidated"


@pytest.mark.parametrize("passing", [False, True])
async def test_unavailable_identity_preserves_grader_result(monkeypatch, tmp_path, passing):
    monkeypatch.setattr(real_run, "_acceptance_digest", lambda directory: None)
    task = load_tasks()[0]
    fixture = tmp_path / "fixture"
    shutil.copytree(task.fixture, fixture)
    if passing:
        shutil.copyfile(task.reference / "calc.py", fixture / "calc.py")
    result = await observed_run(replace(task, fixture=fixture))
    assert result.acceptance_identity == {
        "format": "acceptance-tree-v1",
        "before_sha256": None,
        "after_sha256": None,
        "status": "unavailable",
    }
    assert result.passed is passing
    assert result.acceptance_exit_code == (0 if passing else 1)


@pytest.mark.parametrize("name", ["acceptance_runner.py", "acceptance_pytest.ini"])
def test_identity_includes_owned_grader_bytes(tmp_path, monkeypatch, name):
    import pathlib

    grader_root = tmp_path / "grader"
    grader_root.mkdir()
    original_root = pathlib.Path(real_run.__file__).parent
    for filename in ("acceptance_runner.py", "acceptance_pytest.ini"):
        shutil.copyfile(original_root / filename, grader_root / filename)
    monkeypatch.setattr(real_run, "__file__", str(grader_root / "real_run.py"))
    criteria = load_tasks()[0].acceptance
    before = real_run._acceptance_digest(criteria)
    assert before is not None
    path = grader_root / name
    path.write_bytes(path.read_bytes() + b"\n# Changed grader\n")
    assert real_run._acceptance_digest(criteria) != before


@pytest.mark.parametrize("failure", ["missing", "symlink", "overflow"])
def test_unavailable_criterion_identity_never_becomes_an_empty_digest(tmp_path, failure):
    criteria = tmp_path / "acceptance"
    criteria.mkdir()
    if failure == "missing":
        criteria.rmdir()
    elif failure == "symlink":
        target = tmp_path / "outside.py"
        target.write_text("raise AssertionError\n")
        (criteria / "test_acceptance.py").symlink_to(target)
    else:
        (criteria / "test_acceptance.py").write_bytes(b"x" * 1_000_001)
    assert real_run._acceptance_digest(criteria) is None
