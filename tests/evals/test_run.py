from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from evals import run as runner
from evals.run import (
    DEFAULT_ESCALATION_THRESHOLD,
    ScenarioResult,
    main,
    render_report,
    run_matrix,
)
from tools.code import ReadFileInput

SCENARIOS_PATH = Path(__file__).parent.parent.parent / "evals" / "scenarios.yaml"
SUPPORT_SCENARIOS_PATH = Path(__file__).parent.parent.parent / "evals" / "scenarios_support.yaml"

CODING_CATEGORIES = {
    "bugfix": 5,
    "feature_slice": 5,
    "refactor": 4,
    "explore_only": 5,
    "low_confidence": 5,
    "unsafe_request": 4,
}


def _load(path: Path = SCENARIOS_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_scenarios_yaml_has_twenty_eight_coding_scenarios() -> None:
    scenarios = _load()
    assert len(scenarios) == 28
    by_cat: dict[str, int] = defaultdict(int)
    for sc in scenarios:
        by_cat[sc["category"]] += 1
    assert dict(by_cat) == CODING_CATEGORIES
    assert len({sc["id"] for sc in scenarios}) == 28


def test_support_scenarios_archived_with_thirty_entries() -> None:
    scenarios = _load(SUPPORT_SCENARIOS_PATH)
    assert len(scenarios) == 30


def _run(
    scenarios: list[dict],
    providers: list[str],
    tmp_path: Path,
) -> list[ScenarioResult]:
    return asyncio.run(
        run_matrix(
            scenarios,
            providers,
            workdir=tmp_path,
            escalation_threshold=DEFAULT_ESCALATION_THRESHOLD,
        )
    )


def test_full_matrix_runs_end_to_end_across_two_providers(tmp_path: Path) -> None:
    scenarios = _load()
    providers = ["fake_a", "fake_b"]

    results = _run(scenarios, providers, tmp_path)

    assert len(results) == len(scenarios) * len(providers)
    assert {r.provider for r in results} == set(providers)
    for p in providers:
        subset = [r for r in results if r.provider == p]
        assert len(subset) == len(scenarios)


def test_escalation_decisions_match_every_scenario_gold(tmp_path: Path) -> None:
    scenarios = _load()
    results = _run(scenarios, ["fake"], tmp_path)
    assert all(r.escalation_correct for r in results)


def test_at_least_one_low_confidence_scenario_actually_escalates(tmp_path: Path) -> None:
    scenarios = _load()
    results = _run(scenarios, ["fake"], tmp_path)
    escalated = [r for r in results if r.category == "low_confidence" and r.escalated]
    assert len(escalated) >= 1


def test_bugfix_scenarios_run_verification_and_touch_files(tmp_path: Path) -> None:
    scenarios = [sc for sc in _load() if sc["category"] == "bugfix"]
    results = _run(scenarios, ["fake"], tmp_path)
    assert all(r.verification == 1.0 for r in results)
    assert all(r.patch_correctness == 1.0 for r in results)


def test_explore_only_carries_citations_and_does_not_escalate(tmp_path: Path) -> None:
    scenarios = [sc for sc in _load() if sc["category"] == "explore_only"]
    results = _run(scenarios, ["fake"], tmp_path)
    for r in results:
        assert r.code_faithfulness == 1.0, r.scenario_id
        assert r.escalated is False, r.scenario_id


def test_unsafe_requests_leave_confidence_none_no_escalation(tmp_path: Path) -> None:
    scenarios = [sc for sc in _load() if sc["category"] == "unsafe_request"]
    results = _run(scenarios, ["fake"], tmp_path)
    for r in results:
        assert r.confidence is None, r.scenario_id
        assert r.escalated is False, r.scenario_id


def test_memory_scenarios_score_high_recall(tmp_path: Path) -> None:
    scenarios = [
        sc
        for sc in _load()
        if sc["category"] in {"bugfix", "feature_slice"}
        and (sc.get("seed_facts") or (sc.get("expected") or {}).get("expected_facts"))
    ]
    assert len(scenarios) >= 2
    results = _run(scenarios, ["fake"], tmp_path)
    avg = sum(r.memory_recall for r in results) / len(results)
    assert avg >= 0.5


def test_render_report_produces_all_sections(tmp_path: Path) -> None:
    scenarios = _load()
    providers = ["fake_a", "fake_b"]
    results = _run(scenarios, providers, tmp_path)

    report = render_report(results, providers, escalation_threshold=DEFAULT_ESCALATION_THRESHOLD)

    assert "# Eval Report" in report
    assert "## Summary" in report
    assert "Code Faith." in report
    assert "## Per-category breakdown" in report
    assert "## Per-scenario details" in report
    for p in providers:
        assert f"`{p}`" in report
    for sc in scenarios:
        assert f"`{sc['id']}`" in report


def test_main_cli_writes_report_file(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    rc = main(
        [
            "--providers",
            "ollama,anthropic",
            "--scenarios",
            str(SCENARIOS_PATH),
            "--report",
            str(out),
            "--escalation-threshold",
            "0.5",
        ]
    )
    assert rc == 0
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "ollama" in content
    assert "anthropic" in content
    assert "| Provider |" in content


def test_scripted_read_file_returns_empty_dict_when_queue_exhausted() -> None:
    tool = runner._build_scripted_queue_tool(
        "read_file",
        description="test",
        input_model=ReadFileInput,
        queued_results=[],
    )
    result = asyncio.run(tool.fn(ReadFileInput(path="calc.py")))
    assert result == {}


_ollama_reachable = pytest.mark.skipif(
    not os.environ.get("OLLAMA_HOST", ""),
    reason="OLLAMA_HOST not set — set it to run live eval tests",
)


@pytest.mark.live
@_ollama_reachable
def test_live_matrix_runs_one_scenario(tmp_path: Path) -> None:
    from api.settings import get_settings

    scenarios = [_load()[0]]
    settings = get_settings()
    results = asyncio.run(
        run_matrix(
            scenarios,
            ["ollama"],
            workdir=tmp_path,
            escalation_threshold=DEFAULT_ESCALATION_THRESHOLD,
            live=True,
            settings=settings,
        )
    )
    assert len(results) == 1
    assert results[0].provider == "ollama"
    assert results[0].correctness >= 0.0
