"""CI must fail on broken scripted contracts while retaining an inspectable report."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from evals import run as runner

ROOT = Path(__file__).resolve().parents[2]


def scenario_file(tmp_path, source, *, wrong=False, missing_gold=False):
    scenario = yaml.safe_load((ROOT / "evals" / source).read_text())[0]
    if wrong:
        scenario["expected"]["should_escalate"] = not scenario["expected"]["should_escalate"]
    if missing_gold:
        del scenario["expected"]["should_escalate"]
    path = tmp_path / source
    path.write_text(yaml.safe_dump([scenario]))
    return path


@pytest.mark.parametrize("source", ["scenarios.yaml", "scenarios_support.yaml"])
@pytest.mark.parametrize("wrong", [False, True])
def test_check_exits_for_actual_scripted_gold_and_always_writes_report(tmp_path, source, wrong):
    path = scenario_file(tmp_path, source, wrong=wrong)
    report = tmp_path / "report.md"
    rc = runner.main(
        ["--providers", "ollama", "--scenarios", str(path), "--report", str(report), "--check"]
    )
    assert rc == (1 if wrong else 0)
    content = report.read_text()
    assert source in content and "scripted-contract" in content
    assert "Scripted contract check: " + ("FAIL" if wrong else "PASS") in content
    assert "not independent acceptance checks" in content
    if wrong:
        assert "Escalation mismatch" in content


def test_missing_gold_is_not_an_implicitly_passing_expectation(tmp_path):
    path = scenario_file(tmp_path, "scenarios.yaml", missing_gold=True)
    report = tmp_path / "report.md"
    assert (
        runner.main(
            ["--providers", "ollama", "--scenarios", str(path), "--report", str(report), "--check"]
        )
        == 1
    )
    assert "boolean escalation expectation" in report.read_text()


@pytest.mark.parametrize("empty_source", [False, True])
def test_empty_or_incomplete_matrix_fails_even_without_score_mismatches(
    tmp_path, monkeypatch, empty_source
):
    path = scenario_file(tmp_path, "scenarios.yaml")
    if empty_source:
        path.write_text("[]\n")

    async def missing_results(*args, **kwargs):
        return []

    monkeypatch.setattr(runner, "run_matrix", missing_results)
    report = tmp_path / "report.md"
    assert (
        runner.main(
            ["--providers", "ollama", "--scenarios", str(path), "--report", str(report), "--check"]
        )
        == 1
    )
    assert "Scripted contract check: FAIL" in report.read_text()


def test_check_live_refuses_before_provider_or_settings_access(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "get_settings", lambda: pytest.fail("Must not read live settings"))
    with pytest.raises(SystemExit) as exc:
        runner.main(["--check", "--live", "--report", str(tmp_path / "no-report.md")])
    assert exc.value.code == 2


@pytest.mark.parametrize("fault", ["duplicate", "wrong_provider", "non_scripted"])
def test_check_rejects_wrong_matrix_even_with_correct_escalation(tmp_path, monkeypatch, fault):
    path = scenario_file(tmp_path, "scenarios.yaml")
    original = runner.run_matrix

    async def broken_matrix(*args, **kwargs):
        results = await original(*args, **kwargs)
        assert results[0].escalation_correct
        if fault == "duplicate":
            return results + results
        if fault == "wrong_provider":
            return [replace(results[0], provider="unselected")]
        return [replace(results[0], evaluation_mode=runner.LIVE_SIMULATED_MODE)]

    monkeypatch.setattr(runner, "run_matrix", broken_matrix)
    report = tmp_path / "report.md"
    assert (
        runner.main(
            ["--providers", "ollama", "--scenarios", str(path), "--report", str(report), "--check"]
        )
        == 1
    )
    assert "Scripted contract check: FAIL" in report.read_text()


def test_report_only_mode_keeps_original_exit_contract(tmp_path):
    path = scenario_file(tmp_path, "scenarios.yaml", wrong=True)
    report = tmp_path / "report.md"
    assert (
        runner.main(["--providers", "ollama", "--scenarios", str(path), "--report", str(report)])
        == 0
    )
    assert "Scripted contract check" not in report.read_text()
