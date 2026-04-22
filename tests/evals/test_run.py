from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path

import yaml

from evals import run as runner
from evals.run import (
    DEFAULT_ESCALATION_THRESHOLD,
    ScenarioResult,
    main,
    render_report,
    run_matrix,
)

SCENARIOS_PATH = Path(__file__).parent.parent.parent / "evals" / "scenarios.yaml"


def _load() -> list[dict]:
    with SCENARIOS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── schema sanity ──────────────────────────────────────────────────────────


def test_scenarios_yaml_has_thirty_scenarios_with_six_per_category() -> None:
    scenarios = _load()
    assert len(scenarios) == 30
    by_cat: dict[str, int] = defaultdict(int)
    for sc in scenarios:
        by_cat[sc["category"]] += 1
    assert dict(by_cat) == {
        "grounded_qa": 6,
        "personalization": 6,
        "off_topic": 6,
        "low_confidence": 6,
        "prompt_injection": 6,
    }
    assert len({sc["id"] for sc in scenarios}) == 30


# ─── matrix run ─────────────────────────────────────────────────────────────


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
    # The scripted FakeProvider shares responses across labels, so each
    # scenario's score is identical between the two providers — this is the
    # current state of the harness before step-9 cassettes diverge outputs.
    for p in providers:
        subset = [r for r in results if r.provider == p]
        assert len(subset) == len(scenarios)


def test_escalation_decisions_match_every_scenario_gold(tmp_path: Path) -> None:
    scenarios = _load()
    results = _run(scenarios, ["fake"], tmp_path)
    # Scenarios are authored so the scripted grounder always lands on the
    # gold escalation decision; if this drifts the scenarios or thresholds
    # need to be re-tuned, not the test.
    assert all(r.escalation_correct for r in results)


def test_at_least_one_low_confidence_scenario_actually_escalates(tmp_path: Path) -> None:
    scenarios = _load()
    results = _run(scenarios, ["fake"], tmp_path)
    escalated = [r for r in results if r.category == "low_confidence" and r.escalated]
    # Brief requires at least one escalation trigger to fire on purpose.
    assert len(escalated) >= 1


def test_grounded_scenarios_carry_confidence_and_do_not_escalate(tmp_path: Path) -> None:
    scenarios = [sc for sc in _load() if sc["category"] == "grounded_qa"]
    results = _run(scenarios, ["fake"], tmp_path)
    assert all(r.confidence is not None and r.confidence > 0.5 for r in results)
    assert not any(r.escalated for r in results)


def test_offtopic_and_injection_leave_confidence_none_no_escalation(tmp_path: Path) -> None:
    scenarios = [sc for sc in _load() if sc["category"] in {"off_topic", "prompt_injection"}]
    results = _run(scenarios, ["fake"], tmp_path)
    for r in results:
        assert r.confidence is None, r.scenario_id
        assert r.escalated is False, r.scenario_id


def test_personalization_recall_scores_high_for_seeded_scenarios(tmp_path: Path) -> None:
    scenarios = [sc for sc in _load() if sc["category"] == "personalization"]
    results = _run(scenarios, ["fake"], tmp_path)
    # Every personalization scenario either surfaces its expected fact in
    # the answer or records it via memory_writes; memory_recall averages >0.5.
    avg = sum(r.memory_recall for r in results) / len(results)
    assert avg >= 0.5


# ─── report rendering ───────────────────────────────────────────────────────


def test_render_report_produces_all_sections(tmp_path: Path) -> None:
    scenarios = _load()
    providers = ["fake_a", "fake_b"]
    results = _run(scenarios, providers, tmp_path)

    report = render_report(results, providers, escalation_threshold=DEFAULT_ESCALATION_THRESHOLD)

    assert "# Eval Report" in report
    assert "## Summary" in report
    assert "## Per-category breakdown" in report
    assert "## Per-scenario details" in report
    # Every provider gets a summary row.
    for p in providers:
        assert f"`{p}`" in report
    # Every scenario id appears at least once.
    for sc in scenarios:
        assert f"`{sc['id']}`" in report


def test_main_cli_writes_report_file(tmp_path: Path, monkeypatch: object) -> None:
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


# ─── defensive: scripted search drying up ──────────────────────────────────


def test_scripted_search_returns_empty_when_queue_is_exhausted() -> None:
    tool = runner._build_scripted_search_tool([])
    result = asyncio.run(tool.fn(runner._ScriptedSearchInput(query="anything")))
    assert result == []
