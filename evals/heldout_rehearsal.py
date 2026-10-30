"""Canned-provider, real-tool rehearsal. No network or live-provider selection."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Literal

from evals.heldout_comparison import PLAN_PATH, ComparisonPlan, validate_plan
from evals.real_run import (
    MODES,
    CohortLedger,
    RealTaskResult,
    Task,
    _plan_cohort,
    _validate_report_path,
    load_tasks,
    run_task,
    validate_task,
    write_report,
)
from harness.config import Settings
from providers.base import (
    ChatMessage,
    ChatProvider,
    ProviderResponse,
    TokenUsage,
    ToolCall,
    ToolSpec,
)

Fault = Literal["none", "missing_usage", "wrong_model", "transport"]


class ProtocolStop(ValueError):
    """The remaining fixed cohort must not dispatch."""


class CannedProvider:
    name = "ollama"

    def __init__(self, task: Task, plan: ComparisonPlan, fault: Fault = "none") -> None:
        self.task, self.plan, self.fault = task, plan, fault
        self.calls = 0
        self.closed = False
        self.stop_reason: str | None = None

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        if self.closed or self.stop_reason:
            raise ProtocolStop("Canned provider already stopped")
        self.calls += 1
        if self.fault == "transport":
            self.stop_reason = self.fault
            raise ProtocolStop(f"Canned {self.fault} boundary refusal")
        task = self.task
        question = task.category == "repository_question"
        calls = []
        answer = ""
        if self.calls == 1:
            if question:
                calls = [ToolCall(id="read", name="read_file", arguments={"path": "cache.py"})]
            else:
                files = sorted(p for p in task.reference.iterdir() if p.is_file())
                calls = [
                    ToolCall(
                        id=f"edit-{i}",
                        name="write_file",
                        arguments={"path": p.name, "content": p.read_text()},
                    )
                    for i, p in enumerate(files)
                ]
        elif not question and self.calls == 2:
            calls = [
                ToolCall(
                    id="check",
                    name="run_command",
                    arguments={"argv": self.plan.runtime.project_check_argv},
                )
            ]
        else:
            answer = (
                (task.reference / "answer.json").read_text() if question else "Scripted completion"
            )
        response = ProviderResponse(
            content=answer,
            tool_calls=calls,
            finish_reason="tool_use" if calls else "stop",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            model=self.plan.model,
            latency_ms=0,
        )

        if self.fault == "missing_usage":
            response.usage = TokenUsage(prompt_tokens=None, completion_tokens=5)
        elif self.fault == "wrong_model":
            response.model = "wrong-model"
        return response

    async def aclose(self) -> None:
        self.closed = True


class CheckedProvider:
    """Evaluation boundary: latch failures before returning tool-bearing responses."""

    def __init__(self, provider: ChatProvider, plan: ComparisonPlan) -> None:
        self.provider, self.plan = provider, plan
        self.name = provider.name
        self.stop_reason: str | None = None

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        if self.stop_reason:
            raise ProtocolStop(self.stop_reason)
        try:
            response = await self.provider.chat(
                messages, tools=tools, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as exc:
            self.stop_reason = "transport"
            raise ProtocolStop(self.stop_reason) from exc
        if response.model != self.plan.model:
            self.stop_reason = "wrong_model"
        elif response.usage.prompt_tokens is None or response.usage.completion_tokens is None:
            self.stop_reason = "missing_usage"
        if self.stop_reason:
            raise ProtocolStop(self.stop_reason)
        return response

    async def aclose(self) -> None:
        close = getattr(self.provider, "aclose", None)
        if close is not None:
            await close()


def stop_reason(result: RealTaskResult, plan: ComparisonPlan) -> str | None:
    if result.termination.startswith("error:"):
        return "runtime_or_provider_error"
    if result.model != plan.model:
        return "model_identity"
    if result.prompt_tokens is None or result.completion_tokens is None:
        return "missing_usage"
    if (result.acceptance_identity or {}).get("status") != "matching":
        return "criterion_identity"
    if (result.heldout_preservation or {}).get("status") != "checks_passed":
        return "preservation"
    if result.termination not in (
        "accepted",
        "acceptance_failed",
        "budget_exhausted",
        "iteration_limit",
        "truncated",
        "blocked",
        "incomplete",
    ):
        return "unexpected_terminal_state"
    return None


async def rehearse(
    report: Path, *, plan_path: Path = PLAN_PATH, fault: Fault = "none"
) -> CohortLedger:
    plan = validate_plan(plan_path)
    from evals.heldout_guard import REGISTRATIONS

    manifest = REGISTRATIONS[0].root / "tasks.yaml"
    tasks = load_tasks(manifest)
    _validate_report_path(report, manifest, tasks)
    for task in tasks:
        await validate_task(task)
    ledger, schedule = _plan_cohort(tasks, MODES, plan.repeats)
    settings = Settings(
        _env_file=None,
        _env_prefix="HELDOUT_REHEARSAL_UNUSED_",
        ollama_model=plan.model,
        **plan.runtime.model_dump(),
    )
    results: list[RealTaskResult] = []

    def persist() -> None:
        _validate_report_path(report, manifest, tasks)
        write_report(
            report,
            "scripted held-out rehearsal; no inference",
            results,
            cohort=ledger,
            scripted=True,
        )

    persist()
    try:
        ledger.status = "running"
        for index, (task, mode, attempt) in enumerate(schedule):
            cell = ledger.cells[index]
            provider = CheckedProvider(
                CannedProvider(task, plan, fault if index == 0 else "none"), plan
            )
            try:
                cell.status = "started"
                persist()
                result = await run_task(
                    task, provider=provider, settings=settings, mode=mode, attempt=attempt
                )
                cell.result_index = len(results)
                results.append(result)
                cell.status = "recorded"
            finally:
                await provider.aclose()
            reason = provider.stop_reason or stop_reason(result, plan)
            persist()
            if reason:
                raise ProtocolStop(reason)
        ledger.status = "finished"
    except BaseException as exc:
        ledger.stop(str(exc) if isinstance(exc, ProtocolStop) else type(exc).__name__)
        raise
    finally:
        persist()
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--fault", choices=("none", "missing_usage", "wrong_model", "transport"), default="none"
    )
    args = parser.parse_args()
    asyncio.run(rehearse(args.report, fault=args.fault))
    print("Scripted real-tool rehearsal complete; no model inference ran.")


if __name__ == "__main__":
    main()
