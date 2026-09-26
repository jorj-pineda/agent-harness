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
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from harness.config import Settings, get_settings
from harness.loop import MAX_ITERATIONS_STUB
from harness.prompts import BASE_SYSTEM_PROMPT, PROMPT_VERSION
from harness.runtime import build_registry, run_configured_turn
from harness.state import Session, ToolCallRecord
from memory import FactStore
from providers import create_chat_provider
from providers.base import ChatMessage, ChatProvider, ProviderResponse, ToolSpec
from tools.process import ProcessResult, run_process
from workspace import DisposableWorkspace, disposable_workspace

TASKS_ROOT = Path(__file__).parent / "real_tasks"
MINIMAL_PROMPT = (
    "You are a coding assistant. Complete the user's request using the available tools."
)
Mode = Literal["harness", "minimal"]


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
    provider: str
    model: str
    endpoint: str
    prompt_version: str
    source_revision: str
    passed: bool
    termination: str
    answer: str
    added: tuple[str, ...]
    modified: tuple[str, ...]
    deleted: tuple[str, ...]
    diff: str
    acceptance_exit_code: int
    acceptance_output: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    tool_trace: list[dict[str, Any]]
    runtime_config: dict[str, Any]


class ObservedProvider:
    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider
        self.name = provider.name
        self.model = "unknown"
        self.prompt_tokens = 0
        self.completion_tokens = 0
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
        self.prompt_tokens += response.usage.prompt_tokens or 0
        self.completion_tokens += response.usage.completion_tokens or 0
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
            else "accepted"
            if acceptance.exit_code == 0
            else "acceptance_failed"
        )
        return RealTaskResult(
            task_id=task.id,
            mode=mode,
            provider=provider.name,
            model=observed.model,
            endpoint=settings.ollama_host
            if provider.name == "ollama"
            else f"{provider.name}:default",
            prompt_version="minimal-v1" if mode == "minimal" else PROMPT_VERSION,
            source_revision=_source_revision(copy),
            passed=acceptance.exit_code == 0,
            termination=termination,
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
                "request_timeout_seconds": settings.request_timeout_seconds,
                "require_verification_before_finish": settings.require_verification_before_finish,
                "require_plan_before_edit": settings.require_plan_before_edit,
                "max_files_touched_per_turn": settings.max_files_touched_per_turn,
            },
        )


def _provider(name: str, settings: Settings) -> ChatProvider:
    timeout = float(settings.request_timeout_seconds)
    if name == "ollama":
        return create_chat_provider(
            name,
            host=settings.ollama_host,
            model=settings.ollama_model,
            embed_model=settings.ollama_embed_model,
            timeout_seconds=timeout,
        )
    if name == "anthropic" and settings.anthropic_api_key:
        return create_chat_provider(
            name,
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            timeout_seconds=timeout,
        )
    if name == "openai" and settings.openai_api_key:
        return create_chat_provider(
            name,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout_seconds=timeout,
        )
    raise ValueError(f"Provider {name!r} is unavailable or lacks an API key")


async def _run_cli(args: argparse.Namespace) -> list[RealTaskResult]:
    tasks = load_tasks()
    selected = [task for task in tasks if args.task is None or task.id == args.task]
    if not selected:
        raise ValueError(f"Unknown task: {args.task}")
    for task in selected:
        await validate_task(task)
    if args.validate_only:
        return []
    settings = get_settings()
    provider = _provider(args.provider, settings)
    try:
        return [
            await run_task(task, provider=provider, settings=settings, mode=args.mode)
            for task in selected
        ]
    finally:
        close = getattr(provider, "aclose", None)
        if close is not None:
            await close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-tool coding evaluation on trusted fixtures")
    parser.add_argument("--provider", choices=("ollama", "anthropic", "openai"), default="ollama")
    parser.add_argument("--task", help="Run one task ID; default runs all five")
    parser.add_argument("--mode", choices=("harness", "minimal"), default="harness")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("/tmp/agent-harness-real-eval.json"))
    args = parser.parse_args(argv)
    results = asyncio.run(_run_cli(args))
    if args.validate_only:
        print("Selected task fixtures fail initially and their reference solutions pass.")
        return 0
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {"mode": "real-tools-trusted-fixtures", "results": [asdict(r) for r in results]},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.report} ({sum(r.passed for r in results)}/{len(results)} accepted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
