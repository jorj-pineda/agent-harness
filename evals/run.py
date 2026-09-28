"""Eval runner: scenarios.yaml x providers -> report.md.

Default mode exercises the low-level loop with scripted responses and tools.
`--live` exercises the shared application runtime with a real model, but
still simulates every code tool. Neither mode evaluates working patches.

Coding scenarios script `read_file`, `grep_repo`, `write_file`, and
`run_command` via per-invocation result queues in `tool_results`. Memory
tools use a real tmp `FactStore`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel

from evals.scorers import (
    code_faithfulness,
    correctness,
    escalation,
    memory_recall,
    patch_correctness,
    verification_score,
)
from harness.config import Settings, get_settings
from harness.grounding import Grounder
from harness.loop import run_turn
from harness.prompts import PROMPT_VERSION
from harness.runtime import build_registry, run_configured_turn
from harness.state import Session
from memory import FactStore
from providers import create_chat_provider
from providers.base import (
    ChatMessage,
    ChatProvider,
    FinishReason,
    ProviderResponse,
    ToolCall,
    ToolSpec,
)
from tools import ToolRegistry
from tools.base import Tool, ToolError
from tools.code import (
    GrepRepoInput,
    ReadFileInput,
    RunCommandInput,
    WriteFileInput,
)
from tools.memory import register_memory_tools

log = logging.getLogger(__name__)

DEFAULT_SCENARIOS_PATH = Path(__file__).parent / "scenarios.yaml"
DEFAULT_REPORT_PATH = Path(__file__).parent / "report.md"
DEFAULT_ESCALATION_THRESHOLD = 0.5
DEFAULT_PROVIDERS = "ollama,anthropic"
SCRIPTED_MODE = "scripted-contract"
LIVE_SIMULATED_MODE = "live-model-simulated-tools"

SCRIPTED_TOOL_INPUTS: dict[str, type[BaseModel]] = {
    "read_file": ReadFileInput,
    "grep_repo": GrepRepoInput,
    "write_file": WriteFileInput,
    "run_command": RunCommandInput,
}


@dataclass(frozen=True)
class ScenarioResult:
    """One (scenario x provider) evaluation result."""

    scenario_id: str
    category: str
    provider: str
    code_faithfulness: float
    patch_correctness: float
    verification: float
    correctness: float
    memory_recall: float
    escalation_correct: bool
    escalated: bool
    confidence: float | None
    latency_ms: float
    evaluation_mode: str = SCRIPTED_MODE
    prompt_version: str | None = None
    runtime_config: dict[str, Any] = field(default_factory=dict)


class _ScriptedSearchInput(BaseModel):
    query: str
    k: int = 4
    category: str | None = None


def _build_scripted_queue_tool(
    name: str,
    *,
    description: str,
    input_model: type[BaseModel],
    queued_results: list[Any],
) -> Tool:
    """Tool that pops the next canned result from a queue on each invocation."""

    async def handler(args: BaseModel) -> Any:
        _ = args
        if not queued_results:
            if name == "grep_repo":
                return []
            if name == "run_command":
                return {
                    "argv": [],
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "scripted queue exhausted",
                    "success": False,
                }
            return {}
        return queued_results.pop(0)

    return Tool(
        name=name,
        description=description,
        input_model=input_model,
        fn=handler,
    )


def _build_scripted_search_tool(queued_hits: list[list[dict[str, Any]]]) -> Tool:
    return _build_scripted_queue_tool(
        "search_docs",
        description="Scripted search_docs for eval scenarios.",
        input_model=_ScriptedSearchInput,
        queued_results=queued_hits,
    )


def _build_eval_registry(
    scenario: dict[str, Any],
    *,
    store: FactStore,
    user_id: str,
) -> ToolRegistry:
    registry = ToolRegistry()
    tool_results = scenario.get("tool_results") or {}

    if "search_docs" in tool_results:
        registry.register(_build_scripted_search_tool(list(tool_results["search_docs"])))

    for tool_name, input_model in SCRIPTED_TOOL_INPUTS.items():
        if tool_name in tool_results:
            queue = list(tool_results[tool_name])
            registry.register(
                _build_scripted_queue_tool(
                    tool_name,
                    description=f"Scripted {tool_name} for eval scenarios.",
                    input_model=input_model,
                    queued_results=queue,
                )
            )

    register_memory_tools(registry, store=store, user_id=user_id)
    return registry


def _build_live_simulated_registry(
    scenario: dict[str, Any], *, store: FactStore, user_id: str, workspace_root: str
) -> ToolRegistry:
    """Expose production coding specs without executing any repository tool.

    Missing/exhausted scripts fail explicitly; they must never fall through
    to real execution or manufacture an empty successful tool result.
    """
    results = scenario.get("tool_results") or {}

    def simulate(tool: Tool) -> Tool:
        queue = list(results.get(tool.name) or [])

        async def handler(args: BaseModel) -> Any:
            if not queue:
                raise ToolError(f"Simulated tool {tool.name!r}: no scripted result available")
            return queue.pop(0)

        return replace(tool, fn=handler)

    support_tools = []
    if "search_docs" in results:
        support_tools.append(_build_scripted_search_tool(list(results["search_docs"])))
    return build_registry(
        fact_store=store,
        user_id=user_id,
        workspace_root=workspace_root,
        support_tools=support_tools,
        code_tool_transform=simulate,
    )


class FakeProvider:
    """Scripted ChatProvider: pops pre-canned responses per chat() call."""

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
    return FakeProvider(name, _build_responses(scenario["responses"]))


def _build_live_provider(name: str, settings: Settings) -> ChatProvider:
    timeout = float(settings.request_timeout_seconds)
    match name:
        case "ollama":
            return create_chat_provider(
                "ollama",
                host=settings.ollama_host,
                model=settings.ollama_model,
                embed_model=settings.ollama_embed_model,
                timeout_seconds=timeout,
            )
        case "anthropic":
            if not settings.anthropic_api_key:
                raise SystemExit(f"--live requires ANTHROPIC_API_KEY for provider {name!r}")
            return create_chat_provider(
                "anthropic",
                api_key=settings.anthropic_api_key,
                model=settings.anthropic_model,
                timeout_seconds=timeout,
            )
        case "openai":
            if not settings.openai_api_key:
                raise SystemExit(f"--live requires OPENAI_API_KEY for provider {name!r}")
            return create_chat_provider(
                "openai",
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                timeout_seconds=timeout,
            )
        case _:
            raise SystemExit(f"Unknown provider {name!r} (expected ollama, anthropic, openai)")


def _validate_live_providers(providers: list[str], settings: Settings) -> None:
    for name in providers:
        match name:
            case "ollama":
                continue
            case "anthropic":
                if not settings.anthropic_api_key:
                    raise SystemExit(f"--live requires ANTHROPIC_API_KEY for provider {name!r}")
            case "openai":
                if not settings.openai_api_key:
                    raise SystemExit(f"--live requires OPENAI_API_KEY for provider {name!r}")
            case _:
                raise SystemExit(f"Unknown provider {name!r} (expected ollama, anthropic, openai)")


async def _close_provider(provider: ChatProvider) -> None:
    aclose = getattr(provider, "aclose", None)
    if aclose is not None:
        await aclose()


async def _run_one(
    scenario: dict[str, Any],
    *,
    provider_name: str,
    db_path: Path,
    grounder: Grounder,
    live: bool = False,
    settings: Settings | None = None,
) -> ScenarioResult:
    user_id = scenario.get("user_id") or "eval_user"
    if live:
        if settings is None:
            raise ValueError("settings required when live=True")
        provider: ChatProvider = _build_live_provider(provider_name, settings)
    else:
        provider = _build_provider(provider_name, scenario)

    store = FactStore(db_path)
    try:
        for fact in scenario.get("seed_facts") or []:
            store.add(user_id, fact)

        if live:
            assert settings is not None
            # Never use DEFAULT_WORKSPACE_ROOT: this mode only simulates tools.
            workspace_root = str(db_path.parent.resolve())
            registry = _build_live_simulated_registry(
                scenario, store=store, user_id=user_id, workspace_root=workspace_root
            )
            response = await run_configured_turn(
                settings=settings,
                session=Session(user_id=user_id, workspace_root=workspace_root),
                user_id=user_id,
                message=scenario["user_input"],
                provider=provider,
                fact_store=store,
                registry=registry,
                grounder=grounder,
            )
        else:
            # Preserve the historical low-level contract and its gold scores.
            registry = _build_eval_registry(scenario, store=store, user_id=user_id)
            response = await run_turn(
                session=Session(user_id=user_id),
                user_input=scenario["user_input"],
                provider=provider,
                registry=registry,
                grounder=grounder,
            )
    finally:
        store.close()
        if live:
            await _close_provider(provider)

    expected = scenario.get("expected") or {}
    gold_citations = expected.get("gold_citations") or expected.get("gold_chunks") or []
    return ScenarioResult(
        scenario_id=scenario["id"],
        category=scenario["category"],
        provider=provider_name,
        code_faithfulness=code_faithfulness(response, gold_citations),
        patch_correctness=patch_correctness(response, expected.get("gold_files") or []),
        verification=verification_score(response, expected.get("should_verify")),
        correctness=correctness(response, expected.get("gold_answer") or ""),
        memory_recall=memory_recall(response, expected.get("expected_facts") or []),
        escalation_correct=escalation(response, bool(expected.get("should_escalate"))),
        escalated=response.escalated,
        confidence=response.confidence,
        latency_ms=response.latency_ms,
        evaluation_mode=LIVE_SIMULATED_MODE if live else SCRIPTED_MODE,
        prompt_version=PROMPT_VERSION if live else None,
        runtime_config={
            "max_tool_iterations": settings.max_tool_iterations,
            "max_completion_retries": settings.max_completion_retries,
            "require_plan_before_edit": settings.require_plan_before_edit,
            "require_verification_before_finish": settings.require_verification_before_finish,
            "max_files_touched_per_turn": settings.max_files_touched_per_turn,
            "confidence_escalation_threshold": grounder.escalation_threshold,
        }
        if live and settings
        else {},
    )


async def run_matrix(
    scenarios: list[dict[str, Any]],
    providers: list[str],
    *,
    workdir: Path,
    escalation_threshold: float | None = None,
    live: bool = False,
    settings: Settings | None = None,
) -> list[ScenarioResult]:
    if live and settings is None:
        raise ValueError("settings required when live=True")
    threshold = escalation_threshold
    if threshold is None:
        threshold = (
            settings.confidence_escalation_threshold
            if live and settings
            else DEFAULT_ESCALATION_THRESHOLD
        )
    grounder = Grounder(escalation_threshold=threshold)
    results: list[ScenarioResult] = []
    for scenario_idx, scenario in enumerate(scenarios):
        for provider_name in providers:
            db_path = workdir / f"mem_{scenario_idx:03d}_{provider_name}.db"
            res = await _run_one(
                scenario,
                provider_name=provider_name,
                db_path=db_path,
                grounder=grounder,
                live=live,
                settings=settings,
            )
            results.append(res)
    return results


def _avg(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _pct(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def _summary_row(provider: str, results: list[ScenarioResult]) -> str:
    n = len(results)
    faith = _avg(r.code_faithfulness for r in results)
    patch = _avg(r.patch_correctness for r in results)
    verify = _avg(r.verification for r in results)
    corr = _avg(r.correctness for r in results)
    mem = _avg(r.memory_recall for r in results)
    esc_acc = _pct(sum(1 for r in results if r.escalation_correct), n)
    lat = _avg(r.latency_ms for r in results)
    return (
        f"| `{provider}` | {n} | {faith:.3f} | {patch:.3f} | {verify:.3f} | "
        f"{corr:.3f} | {mem:.3f} | {esc_acc:.3f} | {lat:.1f} |"
    )


def render_report(
    results: list[ScenarioResult],
    providers: list[str],
    *,
    escalation_threshold: float,
    live: bool = False,
) -> str:
    lines: list[str] = []
    lines.append("# Eval Report")
    lines.append("")
    mode = LIVE_SIMULATED_MODE if live else SCRIPTED_MODE
    lines.append(
        f"Coding-agent scenario matrix ({len(results)} results) against "
        f"{len(providers)} providers ({mode}), escalation threshold = "
        f"{escalation_threshold:.2f}."
    )
    lines.append("")
    if live:
        lines.append(
            "Live-model simulated-tool smoke test: real inference with the shared "
            "application prompt, coding tool definitions, memory injection, and policy settings. "
            "Code reads, edits, and commands are simulated; no patch or test suite is executed. "
            "This is not a real coding benchmark. Historical scripted gold escalation labels "
            "may differ from application policy decisions."
        )
        lines.append(f"Prompt version: `{PROMPT_VERSION}`.")
        if results:
            lines.append(f"Runtime settings (non-secret): `{results[0].runtime_config}`.")
    else:
        lines.append(
            "Offline run: every provider replays the same `responses:` "
            "script from `scenarios.yaml`, so per-provider rows match by "
            "construction. This tests the low-level loop contract, without application "
            "prompt injection or scope policy. `--live` adds real inference but still "
            "simulates code tools; neither mode measures real coding success."
        )
    lines.append(
        "Metric limits: Code Faith. measures citation overlap; Patch measures expected-path "
        "recall; Verification measures a trace flag; Correctness measures answer token overlap. "
        "These are contract/smoke metrics, not independent acceptance checks."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Provider | Scenarios | Code Faith. | Patch | Verification | Correctness | "
        "Memory Recall | Escalation Acc. | Mean Latency (ms) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
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
        lines.append(
            "| Provider | Code Faith. | Patch | Verification | Correctness | "
            "Memory Recall | Escalation Acc. |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for provider in providers:
            subset = [r for r in results if r.provider == provider and r.category == category]
            if not subset:
                continue
            faith = _avg(r.code_faithfulness for r in subset)
            patch = _avg(r.patch_correctness for r in subset)
            verify = _avg(r.verification for r in subset)
            corr = _avg(r.correctness for r in subset)
            mem = _avg(r.memory_recall for r in subset)
            esc = _pct(sum(1 for r in subset if r.escalation_correct), len(subset))
            lines.append(
                f"| `{provider}` | {faith:.3f} | {patch:.3f} | {verify:.3f} | "
                f"{corr:.3f} | {mem:.3f} | {esc:.3f} |"
            )
        lines.append("")

    lines.append("## Per-scenario details")
    lines.append("")
    lines.append(
        "| ID | Category | Provider | Faith | Patch | Verify | Correct | MemRec | "
        "EscOK | Escalated | Confidence |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda r: (r.category, r.scenario_id, r.provider)):
        conf = "—" if r.confidence is None else f"{r.confidence:.2f}"
        lines.append(
            f"| `{r.scenario_id}` | {r.category} | `{r.provider}` | "
            f"{r.code_faithfulness:.2f} | {r.patch_correctness:.2f} | {r.verification:.2f} | "
            f"{r.correctness:.2f} | {r.memory_recall:.2f} | "
            f"{'✓' if r.escalation_correct else '✗'} | "
            f"{'yes' if r.escalated else 'no'} | {conf} |"
        )
    lines.append("")
    return "\n".join(lines)


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
        help="Comma-separated provider names, e.g. 'ollama,anthropic,openai'.",
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
        default=None,
        help="Override confidence threshold (default: 0.50 offline; application setting live).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Live-model simulated-tool smoke test (requires API keys / Ollama). "
            "Uses the application runtime but canned code-tool results, not real edits/tests."
        ),
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

    settings: Settings | None = None
    if args.live:
        settings = get_settings()
        _validate_live_providers(providers, settings)

    threshold = args.escalation_threshold
    if threshold is None:
        threshold = (
            settings.confidence_escalation_threshold if settings else DEFAULT_ESCALATION_THRESHOLD
        )

    with tempfile.TemporaryDirectory(prefix="eval_mem_") as tmp_str:
        tmp = Path(tmp_str)
        results = asyncio.run(
            run_matrix(
                scenarios,
                providers,
                workdir=tmp,
                escalation_threshold=threshold,
                live=args.live,
                settings=settings,
            )
        )

    report = render_report(
        results,
        providers,
        escalation_threshold=threshold,
        live=args.live,
    )
    report_path.write_text(report, encoding="utf-8")

    total = len(results)
    esc_ok = sum(1 for r in results if r.escalation_correct)
    print(f"Wrote {report_path} ({total} results, {esc_ok}/{total} escalation decisions correct)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
