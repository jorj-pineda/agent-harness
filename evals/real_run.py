"""Real-tool coding evaluation on five trusted Python fixtures.

The agent edits a disposable copy. Acceptance tests and reference solutions
stay outside that copy. This is a correctness benchmark, not process isolation.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from harness.config import Settings, get_settings
from harness.loop import MAX_ITERATIONS_STUB
from harness.prompts import BASE_SYSTEM_PROMPT, PROMPT_VERSION
from harness.providers import (
    build_configured_provider,
    configured_model,
    configured_provider_names,
    provider_endpoint,
)
from harness.runtime import build_registry, run_configured_turn
from harness.state import Session, ToolCallRecord
from memory import FactStore
from providers.base import ChatMessage, ChatProvider, ProviderResponse, ToolSpec
from tools.process import ProcessResult, run_process
from workspace import DisposableWorkspace, disposable_workspace

TASKS_ROOT = Path(__file__).parent / "real_tasks"
MINIMAL_PROMPT = (
    "You are a coding assistant. Complete the user's request using the available tools."
)
Mode = Literal["harness", "minimal"]
MODES: tuple[Mode, ...] = ("harness", "minimal")


@dataclass(frozen=True)
class Task:
    id: str
    request: str
    fixture: Path
    acceptance: Path
    reference: Path


@dataclass
class RealTaskResult:
    task_id: str
    mode: Mode
    attempt: int
    provider: str
    model: str
    endpoint: str
    prompt_version: str
    source_revision: str
    passed: bool
    termination: str
    completion_status: str
    completion_reason: str | None
    verification_status: str
    answer: str
    added: tuple[str, ...]
    modified: tuple[str, ...]
    deleted: tuple[str, ...]
    diff: str
    acceptance_exit_code: int
    acceptance_output: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    tool_trace: list[dict[str, Any]]
    runtime_config: dict[str, Any]


class ObservedProvider:
    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider
        self.name = provider.name
        self.model = "unknown"
        self.prompt_tokens: int | None = 0
        self.completion_tokens: int | None = 0
        self.finish_reason = "unknown"

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        response = await self.provider.chat(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )
        self.model = response.model
        self.finish_reason = response.finish_reason
        self.prompt_tokens = (
            self.prompt_tokens + response.usage.prompt_tokens
            if self.prompt_tokens is not None and response.usage.prompt_tokens is not None
            else None
        )
        self.completion_tokens = (
            self.completion_tokens + response.usage.completion_tokens
            if self.completion_tokens is not None and response.usage.completion_tokens is not None
            else None
        )
        return response


def load_tasks(path: Path = TASKS_ROOT / "tasks.yaml") -> list[Task]:
    rows = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Task manifest must be a list")
    tasks: list[Task] = []
    for row in rows:
        task_id = str(row["id"])
        if not task_id.replace("_", "").isalnum():
            raise ValueError(f"Invalid task ID: {task_id}")
        fixture = TASKS_ROOT / "fixtures" / str(row["fixture"])
        acceptance = TASKS_ROOT / "acceptance" / task_id
        reference = TASKS_ROOT / "reference" / task_id
        if not fixture.is_dir() or not acceptance.is_dir() or not reference.is_dir():
            raise ValueError(f"Incomplete task: {task_id}")
        tasks.append(Task(task_id, str(row["request"]), fixture, acceptance, reference))
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
    return tasks


def _source_revision(copy: DisposableWorkspace) -> str:
    content = json.dumps(copy.baseline, sort_keys=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _reference_files(task: Task, root: Path) -> None:
    for reference in task.reference.rglob("*"):
        if reference.is_file():
            target = root / reference.relative_to(task.reference)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(reference.read_bytes())


async def acceptance_check(task: Task, root: Path) -> ProcessResult:
    return await run_process(
        [sys.executable, "-m", "pytest", "-q", str(task.acceptance)],
        cwd=root,
        timeout_seconds=30,
        output_limit=12_000,
        env={
            "PYTHONPATH": str(root),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )


async def _visible_check(root: Path) -> ProcessResult:
    return await run_process(
        [sys.executable, "-m", "pytest", "-q", "test_visible.py"],
        cwd=root,
        timeout_seconds=30,
        output_limit=12_000,
        env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONDONTWRITEBYTECODE": "1"},
    )


async def validate_task(task: Task) -> None:
    with disposable_workspace(task.fixture) as baseline:
        initial = await acceptance_check(task, baseline.workspace.root)
        if initial.exit_code == 0:
            raise ValueError(f"Task {task.id} does not fail on its initial fixture")
        visible = await _visible_check(baseline.workspace.root)
        if visible.exit_code == 0:
            raise ValueError(f"Task {task.id} visible check does not fail initially")
    with disposable_workspace(task.fixture) as reference:
        _reference_files(task, reference.workspace.root)
        solved = await acceptance_check(task, reference.workspace.root)
        if solved.exit_code != 0:
            raise ValueError(f"Task {task.id} reference solution fails: {solved.stdout}")
        visible = await _visible_check(reference.workspace.root)
        if visible.exit_code != 0:
            raise ValueError(f"Task {task.id} visible check fails for reference solution")


def _diff(task: Task, copy: DisposableWorkspace, paths: tuple[str, ...]) -> str:
    parts: list[str] = []
    for path in paths:
        original = task.fixture / path
        current = copy.workspace.root / path
        before = original.read_text(encoding="utf-8", errors="replace") if original.exists() else ""
        after = current.read_text(encoding="utf-8", errors="replace") if current.exists() else ""
        parts.extend(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
    return "".join(parts)[:20_000]


async def run_task(
    task: Task,
    *,
    provider: ChatProvider,
    settings: Settings,
    mode: Mode,
    attempt: int = 1,
) -> RealTaskResult:
    observed = ObservedProvider(provider)
    with disposable_workspace(task.fixture) as copy:
        with tempfile.TemporaryDirectory(prefix="agent_harness_eval_memory_") as memory_dir:  # noqa: SIM117 — workspace must outlive memory for grading
            with FactStore(Path(memory_dir) / "memory.db") as store:
                session = Session(user_id="eval", workspace_root=str(copy.workspace.root))
                registry = build_registry(
                    fact_store=store,
                    user_id="eval",
                    workspace_root=str(copy.workspace.root),
                    command_env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                )
                answer = ""
                failure = ""
                completion_status = "incomplete"
                completion_reason: str | None = None
                checked = "not_run"
                start = time.perf_counter()
                try:
                    response = await run_configured_turn(
                        settings=settings,
                        session=session,
                        user_id="eval",
                        message=task.request,
                        provider=observed,
                        fact_store=store,
                        registry=registry,
                        system_prompt=MINIMAL_PROMPT if mode == "minimal" else BASE_SYSTEM_PROMPT,
                    )
                    answer = response.answer
                    completion_status = response.completion_status
                    completion_reason = response.completion_reason
                    checked = response.verification_status
                except Exception as exc:
                    failure = type(exc).__name__
                latency_ms = (time.perf_counter() - start) * 1000
                trace: list[ToolCallRecord] = session.turns[-1].tool_calls if session.turns else []
        acceptance = await acceptance_check(task, copy.workspace.root)
        changes = copy.changes()
        paths = tuple(sorted((*changes.added, *changes.modified, *changes.deleted)))
        termination = (
            f"error:{failure}"
            if failure
            else "iteration_limit"
            if answer == MAX_ITERATIONS_STUB
            else "truncated"
            if observed.finish_reason == "length"
            else completion_status
            if completion_status != "completed"
            else "accepted"
            if acceptance.exit_code == 0
            else "acceptance_failed"
        )
        return RealTaskResult(
            task_id=task.id,
            mode=mode,
            attempt=attempt,
            provider=provider.name,
            model=(
                configured_model(provider.name, settings)
                if observed.model == "unknown"
                and provider.name in configured_provider_names(settings)
                else observed.model
            ),
            endpoint=provider_endpoint(provider.name, settings),
            prompt_version="minimal-v1" if mode == "minimal" else PROMPT_VERSION,
            source_revision=_source_revision(copy),
            passed=acceptance.exit_code == 0,
            termination=termination,
            completion_status=completion_status,
            completion_reason=completion_reason,
            verification_status=checked,
            answer=answer,
            added=changes.added,
            modified=changes.modified,
            deleted=changes.deleted,
            diff=_diff(task, copy, paths),
            acceptance_exit_code=acceptance.exit_code,
            acceptance_output=(acceptance.stdout + acceptance.stderr)[:12_000],
            prompt_tokens=observed.prompt_tokens,
            completion_tokens=observed.completion_tokens,
            latency_ms=latency_ms,
            tool_trace=[call.model_dump(mode="json") for call in trace],
            runtime_config={
                "max_tool_iterations": settings.max_tool_iterations,
                "max_tool_calls_per_turn": settings.max_tool_calls_per_turn,
                "max_turn_wall_seconds": settings.max_turn_wall_seconds,
                "max_completion_tokens_per_turn": settings.max_completion_tokens_per_turn,
                "max_total_tokens_per_turn": settings.max_total_tokens_per_turn,
                "max_context_tokens": settings.max_context_tokens,
                "min_request_output_tokens": settings.min_request_output_tokens,
                "max_identical_tool_calls": settings.max_identical_tool_calls,
                "max_completion_retries": settings.max_completion_retries,
                "project_check_argv": settings.project_check_argv,
                "request_timeout_seconds": settings.request_timeout_seconds,
                "require_verification_before_finish": settings.require_verification_before_finish,
                "require_plan_before_edit": settings.require_plan_before_edit,
                "max_files_touched_per_turn": settings.max_files_touched_per_turn,
                "ollama_num_ctx": settings.ollama_num_ctx,
                "ollama_think": settings.ollama_think,
            },
        )


def summarize(results: list[RealTaskResult]) -> dict[str, Any]:
    """Aggregate raw counts per mode; percentages are left to the reader."""
    summary: dict[str, Any] = {}
    for mode in MODES:
        rows = [r for r in results if r.mode == mode]
        if not rows:
            continue
        tokens = [
            r.prompt_tokens + r.completion_tokens
            for r in rows
            if r.prompt_tokens is not None and r.completion_tokens is not None
        ]
        summary[mode] = {
            "accepted": sum(r.passed for r in rows),
            "attempts": len(rows),
            "per_task": {
                task_id: {
                    "accepted": sum(r.passed for r in rows if r.task_id == task_id),
                    "attempts": sum(r.task_id == task_id for r in rows),
                }
                for task_id in sorted({r.task_id for r in rows})
            },
            "terminations": dict(sorted(Counter(r.termination for r in rows).items())),
            "false_completion": sum(
                r.completion_status == "completed" and not r.passed for r in rows
            ),
            "median_latency_s": round(statistics.median(r.latency_ms for r in rows) / 1000, 1),
            "median_tool_calls": statistics.median(len(r.tool_trace) for r in rows),
            "median_total_tokens": statistics.median(tokens) if tokens else None,
            "unreported_token_attempts": len(rows) - len(tokens),
        }
    return summary


def summary_markdown(summary: dict[str, Any]) -> str:
    modes = list(summary)
    task_ids = sorted({task for mode in modes for task in summary[mode]["per_task"]})
    lines = ["| Task | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for task_id in task_ids:
        cells = [
            "{accepted}/{attempts}".format(
                **summary[mode]["per_task"].get(task_id, {"accepted": 0, "attempts": 0})
            )
            for mode in modes
        ]
        lines.append(f"| {task_id} | " + " | ".join(cells) + " |")
    lines.append(
        "| **total** | "
        + " | ".join(f"**{summary[m]['accepted']}/{summary[m]['attempts']}**" for m in modes)
        + " |"
    )
    return "\n".join(lines)


def _harness_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def write_report(path: Path, started: str, results: list[RealTaskResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "mode": "real-tools-trusted-fixtures",
                "started_at": started,
                "harness_revision": _harness_revision(),
                "sampling": {"temperature": 0.0},
                "summary": summarize(results),
                "results": [asdict(r) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


async def _run_cli(args: argparse.Namespace, started: str = "") -> list[RealTaskResult]:
    tasks = load_tasks()
    selected = [task for task in tasks if args.task is None or task.id == args.task]
    if not selected:
        raise ValueError(f"Unknown task: {args.task}")
    for task in selected:
        await validate_task(task)
    if args.validate_only:
        return []
    settings = get_settings()
    provider = build_configured_provider(args.provider, settings)
    modes: tuple[Mode, ...] = MODES if args.mode == "both" else (args.mode,)
    results: list[RealTaskResult] = []
    try:
        for attempt in range(1, args.repeats + 1):
            for task in selected:
                # Alternate mode order so neither mode always runs on a freshly loaded model.
                ordered = modes if (attempt + selected.index(task)) % 2 else modes[::-1]
                for mode in ordered:
                    result = await run_task(
                        task, provider=provider, settings=settings, mode=mode, attempt=attempt
                    )
                    results.append(result)
                    write_report(args.report, started, results)
                    print(
                        f"{task.id} {mode} #{attempt}: {result.termination} "
                        f"({result.latency_ms / 1000:.0f}s, {len(result.tool_trace)} tools)",
                        flush=True,
                    )
    finally:
        close = getattr(provider, "aclose", None)
        if close is not None:
            await close()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-tool coding evaluation on trusted fixtures")
    parser.add_argument(
        "--provider",
        choices=("ollama", "anthropic", "openai", "openai_compatible"),
        default="ollama",
    )
    parser.add_argument("--task", help="Run one task ID; default runs all five")
    parser.add_argument("--mode", choices=("harness", "minimal", "both"), default="harness")
    parser.add_argument("--repeats", type=int, default=1, help="Attempts per task and mode")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("/tmp/agent-harness-real-eval.json"))
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    started = datetime.now(UTC).isoformat(timespec="seconds")
    results = asyncio.run(_run_cli(args, started))
    if args.validate_only:
        print("Selected task fixtures fail initially and their reference solutions pass.")
        return 0
    write_report(args.report, started, results)
    summary = summarize(results)
    print(summary_markdown(summary))
    print(f"Wrote {args.report} ({sum(r.passed for r in results)}/{len(results)} accepted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
