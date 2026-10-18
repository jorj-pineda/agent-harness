import json
from dataclasses import replace

import pytest

from evals.real_run import acceptance_check, load_tasks, run_task, write_report
from harness.config import Settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from workspace import disposable_workspace


@pytest.mark.parametrize("injection", ["module", "package", "conftest", "pytest_ini", "pyproject"])
async def test_workspace_cannot_select_pytest_or_grader_configuration(injection):
    task = load_tasks()[0]
    with disposable_workspace(task.fixture) as copy:
        root = copy.workspace.root
        if injection == "module":
            (root / "pytest.py").write_text("raise SystemExit(0)\n")
        elif injection == "package":
            (root / "pytest").mkdir()
            (root / "pytest" / "__init__.py").write_text("raise SystemExit(0)\n")
            (root / "pytest" / "__main__.py").write_text("raise SystemExit(0)\n")
        elif injection == "conftest":
            (root / "conftest.py").write_text("raise RuntimeError('workspace-hook-loaded')\n")
        elif injection == "pytest_ini":
            (root / "pytest.ini").write_text(
                "[pytest]\naddopts = --workspace-option-must-not-load\n"
            )
        else:
            (root / "pyproject.toml").write_text(
                '[tool.pytest.ini_options]\naddopts = "--workspace-option-must-not-load"\n'
            )
        broken = await acceptance_check(task, root)
        assert broken.exit_code == 1, broken.stdout + broken.stderr
        assert "test_division" in broken.stdout
        (root / "calc.py").write_bytes((task.reference / "calc.py").read_bytes())
        solved = await acceptance_check(task, root)
        assert solved.exit_code == 0, solved.stdout + solved.stderr
        assert "2 passed" in solved.stdout
        assert (
            "workspace-hook-loaded"
            not in broken.stdout + broken.stderr + solved.stdout + solved.stderr
        )
        assert not (root / ".pytest_cache").exists()


async def test_acceptance_inside_editable_workspace_is_rejected():
    task = load_tasks()[0]
    with disposable_workspace(task.fixture) as copy:
        result = await acceptance_check(
            replace(task, acceptance=copy.workspace.root), copy.workspace.root
        )
    assert result.exit_code != 0
    assert "Authoritative acceptance must be outside" in result.stderr


async def test_scripted_false_visible_pass_cannot_replace_external_acceptance(tmp_path):
    task = load_tasks()[0]
    original = (task.fixture / "calc.py").read_bytes()
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="shadow",
                    name="write_file",
                    arguments={"path": "pytest.py", "content": "raise SystemExit(0)\n"},
                )
            ]
        ),
        make_response(
            tool_calls=[
                ToolCall(
                    id="check",
                    name="run_command",
                    arguments={"argv": ["python", "-m", "pytest", "-q"]},
                )
            ]
        ),
        make_response(content="Claimed complete"),
    )
    result = await run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.verification_status == "passed"
    assert not result.passed and result.acceptance_exit_code == 1
    assert "test_division" in result.acceptance_output
    assert result.acceptance_policy == "owned-pytest-v1"
    report = tmp_path / "report.json"
    write_report(report, "offline boundary validation", [result])
    assert json.loads(report.read_text())["results"][0]["acceptance_policy"] == "owned-pytest-v1"
    assert (task.fixture / "calc.py").read_bytes() == original


@pytest.mark.parametrize(
    "stdout,stderr,stdout_truncated,stderr_truncated,truncated",
    [
        ("", "", False, False, False),
        ("x" * 12_000, "", False, False, False),
        ("x" * 12_001, "", False, False, True),
        ("x" * 7_000, "y" * 5_001, False, False, True),
        ("prefix", "", True, False, True),
        ("", "prefix", False, True, True),
    ],
    ids=["empty", "exact-report-cap", "over-report-cap", "combined-overflow", "stdout", "stderr"],
)
async def test_report_retains_acceptance_output_coverage(
    monkeypatch, tmp_path, stdout, stderr, stdout_truncated, stderr_truncated, truncated
):
    from evals import real_run
    from tools.process import ProcessResult

    async def scripted_grader(task, root):
        return ProcessResult([], 1, stdout, stderr, stdout_truncated, stderr_truncated)

    monkeypatch.setattr(real_run, "acceptance_check", scripted_grader)
    provider = ScriptedProvider()
    provider.script(make_response(content="No patch"))
    result = await run_task(
        load_tasks()[0], provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.acceptance_output == (stdout + stderr)[:12_000]
    assert result.acceptance_output_truncated is truncated
    assert result.acceptance_exit_code == 1 and not result.passed
    path = tmp_path / "report.json"
    write_report(path, "scripted output coverage", [result])
    row = json.loads(path.read_text())["results"][0]
    assert row["acceptance_output_truncated"] is truncated
    # Legacy/custom rows do not acquire an invented complete-output assertion.
    row.pop("acceptance_output_truncated")
    assert real_run.RealTaskResult(**row).acceptance_output_truncated is None


async def test_real_noisy_grader_retains_rejection_and_explicit_incomplete_output(tmp_path):
    acceptance = tmp_path / "acceptance"
    acceptance.mkdir()
    (acceptance / "test_acceptance.py").write_text(
        'def test_noisy_failure():\n    print("x" * 16000)\n    assert False\n'
    )
    task = replace(load_tasks()[0], acceptance=acceptance)
    provider = ScriptedProvider()
    provider.script(make_response(content="No patch"))
    result = await run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert not result.passed and result.acceptance_exit_code == 1
    assert result.acceptance_output_truncated is True
    assert len(result.acceptance_output) <= 12_000
    assert "test_noisy_failure" in result.acceptance_output
