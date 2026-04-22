"""Eval runner: scenarios.yaml x providers -> report.md.

Drives `harness.loop.run_turn` directly — skips the FastAPI layer so this
step doesn't block on step 10's merge. Each scenario runs against every
provider passed on the CLI.

Provider handling:
    The per-provider cassette work lives on `feat/provider-cassettes`
    (step 9). Until those cassettes land, every named provider replays
    the same scripted responses from `scenarios.yaml` through a
    `FakeProvider` whose `name` attribute is the provider label. The
    report columns therefore match until real cassettes diverge the
    outputs — swapping them in is a single call-site change inside
    `_build_provider`.

Tools:
    * `search_docs` is scripted: it dequeues pre-canned hits from the
      scenario so retrieval is deterministic and the evals run offline.
    * `remember_fact` / `recall_facts` use a real tmp `FactStore`, so
      personalization recall actually exercises the memory layer.

Output:
    Markdown report at `evals/report.md` (configurable). Sections:
    summary averages, per-category breakdown, per-scenario details.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel

from evals.scorers import correctness, escalation, faithfulness, memory_recall
from harness.grounding import Grounder
from harness.loop import run_turn
from harness.state import Session
from memory import FactStore
from providers.base import (
    ChatMessage,
    FinishReason,
    ProviderResponse,
    ToolCall,
    ToolSpec,
)
from tools import ToolRegistry
from tools.base import Tool
from tools.memory import register_memory_tools

log = logging.getLogger(__name__)

DEFAULT_SCENARIOS_PATH = Path(__file__).parent / "scenarios.yaml"
DEFAULT_REPORT_PATH = Path(__file__).parent / "report.md"
DEFAULT_ESCALATION_THRESHOLD = 0.5
DEFAULT_PROVIDERS = "ollama,anthropic"


@dataclass(frozen=True)
class ScenarioResult:
    """One (scenario x provider) evaluation result."""

    scenario_id: str
    category: str
    provider: str
    faithfulness: float
    correctness: float
    memory_recall: float
    escalation_correct: bool
    escalated: bool
    confidence: float | None
    latency_ms: float


# ─── Scripted tool + provider ──────────────────────────────────────────────


class _ScriptedSearchInput(BaseModel):
    """Mirror of the real search_docs signature so scripted calls validate."""

    query: str
    k: int = 4
    category: str | None = None


def _build_scripted_search_tool(queued_hits: list[list[dict[str, Any]]]) -> Tool:
    """A search_docs tool that pops the next list of hits from a queue.

    `queued_hits` is mutated in place — each invocation consumes one entry.
    Running dry returns `[]` so scenarios that retry don't explode on an
    IndexError; the grounder then treats it as a zero-hit retrieval.
    """

    async def search_docs(args: _ScriptedSearchInput) -> list[dict[str, Any]]:
        if not queued_hits:
            return []
        return queued_hits.pop(0)

    return Tool(
        name="search_docs",
        description="Scripted search_docs for eval scenarios.",
        input_model=_ScriptedSearchInput,
        fn=search_docs,
    )


class FakeProvider:
    """Scripted ChatProvider: pops pre-canned responses per chat() call.

    `name` is set to the provider label so `TurnResponse.provider` matches
    whatever the scenario matrix is running against.
    """

    def __init__(self, name: str, responses: Iterable[ProviderResponse]) -> None:
        self.name = name
        self._queue: list[ProviderResponse] = list(responses)

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        if not self._queue:
            raise RuntimeError(
                f"FakeProvider {self.name!r} ran out of scripted responses; "
                "scenario has fewer responses than the harness requested."
            )
        return self._queue.pop(0)


def _build_responses(raw: list[dict[str, Any]]) -> list[ProviderResponse]:
    """Compile YAML `responses:` entries into `ProviderResponse` objects.

    Tool-call ids are synthesised (`t{resp}_{tool}`) so scenario authors
    don't have to manage them by hand; `run_turn` just needs them to be
    unique within a turn.
    """
    out: list[ProviderResponse] = []
    for resp_idx, entry in enumerate(raw):
        tool_calls: list[ToolCall] = []
        for tc_idx, tc in enumerate(entry.get("tool_calls") or []):
            tool_calls.append(
                ToolCall(
                    id=f"t{resp_idx}_{tc_idx}",
                    name=tc["name"],
                    arguments=tc.get("arguments") or {},
                )
            )
        finish: FinishReason = "tool_use" if tool_calls else "stop"
        out.append(
            ProviderResponse(
                content=entry.get("content", "") or "",
                tool_calls=tool_calls,
                finish_reason=finish,
                model="scripted",
                latency_ms=0.0,
            )
        )
    return out


def _build_provider(name: str, scenario: dict[str, Any]) -> FakeProvider:
    # Per-provider overrides land here when cassettes arrive; until then
    # every provider shares the scenario's single `responses:` script.
    return FakeProvider(name, _build_responses(scenario["responses"]))


# ─── Scenario execution ────────────────────────────────────────────────────


async def _run_one(
    scenario: dict[str, Any],
    *,
    provider_name: str,
    db_path: Path,
    grounder: Grounder,
) -> ScenarioResult:
    user_id = scenario.get("user_id") or "eval_user"
    store = FactStore(db_path)
    try:
        for fact in scenario.get("seed_facts") or []:
            store.add(user_id, fact)

        registry = ToolRegistry()
        queued_hits = list((scenario.get("tool_results") or {}).get("search_docs") or [])
        registry.register(_build_scripted_search_tool(queued_hits))
        register_memory_tools(registry, store=store, user_id=user_id)

        provider = _build_provider(provider_name, scenario)
        session = Session(user_id=user_id)
        response = await run_turn(
            session=session,
            user_input=scenario["user_input"],
            provider=provider,
            registry=registry,
            grounder=grounder,
        )
    finally:
        store.close()

    expected = scenario.get("expected") or {}
    return ScenarioResult(
        scenario_id=scenario["id"],
        category=scenario["category"],
        provider=provider_name,
        faithfulness=faithfulness(response, expected.get("gold_chunks") or []),
        correctness=correctness(response, expected.get("gold_answer") or ""),
        memory_recall=memory_recall(response, expected.get("expected_facts") or []),
        escalation_correct=escalation(response, bool(expected.get("should_escalate"))),
        escalated=response.escalated,
        confidence=response.confidence,
        latency_ms=response.latency_ms,
    )


async def run_matrix(
    scenarios: list[dict[str, Any]],
    providers: list[str],
    *,
    workdir: Path,
    escalation_threshold: float = DEFAULT_ESCALATION_THRESHOLD,
) -> list[ScenarioResult]:
    """Run every (scenario x provider) pair and return results in order."""
    grounder = Grounder(escalation_threshold=escalation_threshold)
    results: list[ScenarioResult] = []
    for scenario_idx, scenario in enumerate(scenarios):
        for provider_name in providers:
            db_path = workdir / f"mem_{scenario_idx:03d}_{provider_name}.db"
            res = await _run_one(
                scenario,
                provider_name=provider_name,
                db_path=db_path,
                grounder=grounder,
            )
            results.append(res)
    return results


# ─── Report rendering ──────────────────────────────────────────────────────


def _avg(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _pct(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def _summary_row(provider: str, results: list[ScenarioResult]) -> str:
    n = len(results)
    faith = _avg(r.faithfulness for r in results)
    corr = _avg(r.correctness for r in results)
    mem = _avg(r.memory_recall for r in results)
    esc_acc = _pct(sum(1 for r in results if r.escalation_correct), n)
    lat = _avg(r.latency_ms for r in results)
    return (
        f"| `{provider}` | {n} | {faith:.3f} | {corr:.3f} | {mem:.3f} | {esc_acc:.3f} | {lat:.1f} |"
    )


def render_report(
    results: list[ScenarioResult],
    providers: list[str],
    *,
    escalation_threshold: float,
) -> str:
    lines: list[str] = []
    lines.append("# Eval Report")
    lines.append("")
    lines.append(
        f"Full scenario matrix ({len(results)} results) against "
        f"{len(providers)} providers, escalation threshold = "
        f"{escalation_threshold:.2f}."
    )
    lines.append("")
    lines.append(
        "Providers run through a scripted `FakeProvider` until cassette "
        "infrastructure from `feat/provider-cassettes` lands; until then "
        "per-provider rows share the same scripted responses and differ "
        "only by label."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Provider | Scenarios | Faithfulness | Correctness | Memory Recall | "
        "Escalation Acc. | Mean Latency (ms) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for provider in providers:
        subset = [r for r in results if r.provider == provider]
        if subset:
            lines.append(_summary_row(provider, subset))
    lines.append("")

    categories = sorted({r.category for r in results})
    lines.append("## Per-category breakdown")
    lines.append("")
    for category in categories:
        lines.append(f"### `{category}`")
        lines.append("")
        lines.append("| Provider | Faithfulness | Correctness | Memory Recall | Escalation Acc. |")
        lines.append("|---|---|---|---|---|")
        for provider in providers:
            subset = [r for r in results if r.provider == provider and r.category == category]
            if not subset:
                continue
            faith = _avg(r.faithfulness for r in subset)
            corr = _avg(r.correctness for r in subset)
            mem = _avg(r.memory_recall for r in subset)
            esc = _pct(sum(1 for r in subset if r.escalation_correct), len(subset))
            lines.append(f"| `{provider}` | {faith:.3f} | {corr:.3f} | {mem:.3f} | {esc:.3f} |")
        lines.append("")

    lines.append("## Per-scenario details")
    lines.append("")
    lines.append(
        "| ID | Category | Provider | Faith | Correct | MemRec | EscOK | Escalated | Confidence |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda r: (r.category, r.scenario_id, r.provider)):
        conf = "—" if r.confidence is None else f"{r.confidence:.2f}"
        lines.append(
            f"| `{r.scenario_id}` | {r.category} | `{r.provider}` | "
            f"{r.faithfulness:.2f} | {r.correctness:.2f} | {r.memory_recall:.2f} | "
            f"{'✓' if r.escalation_correct else '✗'} | "
            f"{'yes' if r.escalated else 'no'} | {conf} |"
        )
    lines.append("")
    return "\n".join(lines)


# ─── CLI ───────────────────────────────────────────────────────────────────


def _load_scenarios(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of scenarios, got {type(data).__name__}")
    return cast(list[dict[str, Any]], data)


def _parse_providers(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.run",
        description="Run the agent-harness eval scenario matrix.",
    )
    parser.add_argument(
        "--providers",
        default=DEFAULT_PROVIDERS,
        help="Comma-separated provider names, e.g. 'ollama,anthropic'.",
    )
    parser.add_argument(
        "--scenarios",
        default=str(DEFAULT_SCENARIOS_PATH),
        help="Path to scenarios YAML file.",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT_PATH),
        help="Destination path for the generated markdown report.",
    )
    parser.add_argument(
        "--escalation-threshold",
        type=float,
        default=DEFAULT_ESCALATION_THRESHOLD,
        help="Confidence threshold below which a turn is escalated (0-1).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    providers = _parse_providers(args.providers)
    if not providers:
        raise SystemExit("--providers must list at least one provider name")

    scenarios = _load_scenarios(Path(args.scenarios))
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="eval_mem_") as tmp_str:
        tmp = Path(tmp_str)
        results = asyncio.run(
            run_matrix(
                scenarios,
                providers,
                workdir=tmp,
                escalation_threshold=args.escalation_threshold,
            )
        )

    report = render_report(results, providers, escalation_threshold=args.escalation_threshold)
    report_path.write_text(report, encoding="utf-8")

    total = len(results)
    esc_ok = sum(1 for r in results if r.escalation_correct)
    print(f"Wrote {report_path} ({total} results, {esc_ok}/{total} escalation decisions correct)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
