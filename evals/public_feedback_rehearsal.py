"""Mandatory-mock native inputs and real-tool rehearsal; no live transport option."""

from __future__ import annotations

import json
import stat
from collections.abc import Callable, Mapping
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field

from evals.heldout_command_environment import environment_bytes
from evals.heldout_comparison import ProtocolModel
from evals.heldout_guard import _read_regular
from evals.heldout_inputs import (
    _save,
    check_dependency_selection,
    check_runtime_selection,
    normalized_input,
)
from evals.heldout_native_evidence import NativeEvidenceTransport
from evals.heldout_settings import check_settings, settings_bytes
from evals.public_envelope import _digest, public_settings
from evals.public_feedback import PLAN_SHA256, FeedbackCell, validate_feedback_plan
from evals.public_feedback_workspaces import (
    FeedbackCellWorkspace,
    FeedbackFixture,
    _selection,
    prepare_feedback_cell,
)
from evals.public_inputs import Digest, PublicRuntime
from evals.public_workspaces import _destination
from evals.real_run import _harness_revision, _task_command_environment, run_task
from providers.base import ChatMessage, ProviderResponse, ToolSpec
from providers.ollama import OllamaProvider


class RehearsalCell(ProtocolModel):
    cell: FeedbackCell
    workspace_root: str
    files: dict[str, Digest] = Field(min_length=20, max_length=20)


class FeedbackRehearsal(ProtocolModel):
    format: Literal["public-feedback-mock-rehearsal-v1"] = "public-feedback-mock-rehearsal-v1"
    plan_sha256: Digest
    runtime: PublicRuntime
    settings_sha256: Digest
    command_environment_sha256: Digest
    runtime_inventory_sha256: Digest
    dependency_files_sha256: Digest
    report_sha256: Digest
    cells: list[RehearsalCell] = Field(min_length=8, max_length=8)
    generation_forwarded: Literal[0] = 0
    inference_authorized: Literal[False] = False
    evidence_label: Literal["scripted-real-tools-rehearsal"] = "scripted-real-tools-rehearsal"


class _CapturedProvider:
    name = "ollama"

    def __init__(
        self, adapter: OllamaProvider, check: Callable[[], None], capture: Callable[[bytes], None]
    ) -> None:
        self.adapter, self.check, self.capture = adapter, check, capture
        self.captured = False

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        self.check()
        if not self.captured:
            self.capture(normalized_input(messages, tools, temperature, max_tokens))
            self.captured = True
        response = await self.adapter.chat(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )
        self.check()
        return response


async def rehearse_feedback(
    directory: Path,
    *,
    command_env: Mapping[str, str],
    runtime_inventory: Path,
    runtime_sha256: str,
    dependency_manifest: Path,
    dependency_sha256: str,
) -> FeedbackRehearsal:
    """Eight fresh consumed cells, only MockTransport, with an index written last."""
    plan = validate_feedback_plan()
    observed = _harness_revision()
    runtime = PublicRuntime.model_validate(observed)
    selected_env = _task_command_environment(command_env)
    if selected_env is None:
        raise ValueError("Select the rehearsal command replacement environment")
    settings = public_settings()
    expected_settings = settings_bytes(settings)
    check_runtime_selection(runtime_inventory, runtime_sha256)
    check_dependency_selection(dependency_manifest, dependency_sha256)
    destination = _destination(directory)
    destination.mkdir(mode=0o700, exist_ok=False)
    saved: dict[str, str] = {}
    outer: set[str] = set()

    def save(name: str, raw: bytes) -> str:
        digest = _save(destination / name, raw)
        saved[name] = digest
        outer.add(Path(name).parts[0])
        return digest

    settings_sha = save("settings.json", expected_settings)
    env_sha = save("command-environment.json", environment_bytes(selected_env))
    for source, digest, name in (
        (runtime_inventory, runtime_sha256, "runtime-inventory.json"),
        (dependency_manifest, dependency_sha256, "dependency-files.json"),
    ):
        raw, mode = _read_regular(source, 2_000_000)
        if mode != 0o600 or _digest(raw) != digest:
            raise ValueError("Selected rehearsal input changed before preservation")
        save(name, raw)

    def check() -> None:
        if (
            destination.is_symlink()
            or _destination(directory) != destination
            or stat.S_IMODE(destination.stat().st_mode) != 0o700
        ):
            raise ValueError("Rehearsal output container changed")
        validate_feedback_plan()
        if _harness_revision() != observed:
            raise ValueError("Rehearsal source runtime changed")
        check_settings(settings, expected_settings)
        check_runtime_selection(destination / "runtime-inventory.json", runtime_sha256)
        check_dependency_selection(destination / "dependency-files.json", dependency_sha256)
        for name, digest in saved.items():
            raw, mode = _read_regular(destination / name, 2_000_000)
            if mode != 0o600 or _digest(raw) != digest:
                raise ValueError("Rehearsal captured evidence changed")
        if {p.name for p in islice(destination.iterdir(), len(outer) + 1)} != outer:
            raise ValueError("Rehearsal output inventory changed")

    rows: list[RehearsalCell] = []
    results = []
    for cell in plan.cells:
        check()
        container = destination / f"cell-{cell.ordinal:02d}"
        receipt = prepare_feedback_cell(cell.ordinal, container)
        outer.add(container.name)
        _, task = _selection(cell.ordinal)
        prefix = f"{cell.ordinal:02d}"
        files: dict[str, str] = {}
        initial, _ = _read_regular(container / "workspace/test_visible.py", 16_000)
        files[f"{prefix}-initial-visible.py"] = save(f"{prefix}-initial-visible.py", initial)
        reference_raw, _ = _read_regular(task.reference / "pagination.py", 64_000)
        reference = reference_raw.decode("utf-8")
        defective = reference.replace("type(page) is not int", "not isinstance(page, int)")
        defective = defective.replace(
            "type(page_size) is not int", "not isinstance(page_size, int)"
        )
        requests = 0
        expected_native: dict[str, str] = {}

        def capture(raw: bytes, label: str = prefix, selected: dict[str, str] = files) -> None:
            name = f"{label}-normalized.json"
            selected[name] = save(name, raw)

        def intercepted(
            request: httpx.Request,
            *,
            arm: str = cell.arm,
            broken: str = defective,
            solved: str = reference,
            native_pins: dict[str, str] = expected_native,
        ) -> httpx.Response:
            nonlocal requests
            check()
            requests += 1
            if request.method != "POST" or request.url.path != "/api/chat" or requests > 4:
                raise ValueError("Unexpected rehearsal native request")
            payload = json.loads(request.content)
            names = {t["function"]["name"] for t in payload["tools"]}
            if not {"read_file", "write_file", "run_project_check"} <= names:
                raise ValueError("Rehearsal selected tools unavailable")
            calls = []
            if requests == 1:
                calls = [
                    ("read_file", {"path": name}) for name in ("pagination.py", "test_visible.py")
                ]
            elif requests in (2, 3):
                if requests == 3:
                    first_check = json.loads(payload["messages"][-1]["content"])
                    if first_check["exit_code"] != (1 if arm == "bool_check" else 0):
                        raise ValueError(
                            "Rehearsal did not observe the selected visible-check outcome"
                        )
                calls = [
                    (
                        "write_file",
                        {
                            "path": "pagination.py",
                            "content": broken if requests == 2 else solved,
                        },
                    ),
                    ("run_project_check", {}),
                ]
            else:
                if json.loads(payload["messages"][-1]["content"])["exit_code"] != 0:
                    raise ValueError("Rehearsal reference check did not pass")
            response = httpx.Response(
                200,
                json={
                    "model": plan.model,
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 100,
                    "eval_count": 10,
                    "message": {
                        "role": "assistant",
                        "content": "Scripted reference check passed." if not calls else "",
                        "tool_calls": [
                            {"function": {"name": name, "arguments": args}} for name, args in calls
                        ],
                    },
                },
            )
            native_pins[f"request-{requests:02d}.request.bin"] = _digest(request.content)
            native_pins[f"request-{requests:02d}.response.bin"] = _digest(response.content)
            return response

        native_dir = destination / f"native-{prefix}"
        native = NativeEvidenceTransport(httpx.MockTransport(intercepted), native_dir)
        outer.add(native_dir.name)
        adapter = OllamaProvider(
            host=settings.ollama_host,
            model=plan.model,
            embed_model="unused",
            timeout_seconds=settings.request_timeout_seconds,
            num_ctx=settings.ollama_num_ctx,
            think=settings.ollama_think,
            transport=native,
        )
        try:
            result = await run_task(
                task,
                provider=_CapturedProvider(adapter, check, capture),
                settings=settings,
                mode="harness",
                attempt=cell.repeat,
                feedback_cell=FeedbackCellWorkspace(ordinal=cell.ordinal, directory=container),
                command_env=selected_env,
            )
            native_failed = native.failed
        finally:
            await adapter.aclose()
        files[f"{prefix}-result.json"] = save(
            f"{prefix}-result.json", json.dumps(asdict(result), sort_keys=True).encode()
        )
        check()
        if (
            requests != 4
            or native.requests != 4
            or native_failed
            or not native.closed
            or not result.passed
            or result.completion_status != "completed"
            or result.verification_status != "passed"
            or result.runtime_config["public_feedback_preservation"]["status"] != "checks_passed"
        ):
            raise ValueError("Mandatory-mock reference rehearsal did not complete")
        final, _ = _read_regular(container / "workspace/test_visible.py", 16_000)
        implementation, _ = _read_regular(container / "workspace/pagination.py", 64_000)
        if final != initial or receipt.cell != cell:
            raise ValueError("Rehearsal visible test or cell identity changed")
        for label, raw in (
            ("final-visible.py", final),
            ("final-implementation.py", implementation),
            ("runtime-config.json", json.dumps(result.runtime_config, sort_keys=True).encode()),
        ):
            name = f"{prefix}-{label}"
            files[name] = save(name, raw)
        if {p.name for p in islice(container.iterdir(), 4)} != {
            "workspace",
            "prepared.json",
            "admitted.json",
        }:
            raise ValueError("Rehearsal cell container inventory changed")
        for name in ("prepared.json", "admitted.json"):
            relative = f"{container.name}/{name}"
            raw, mode = _read_regular(destination / relative, 4096)
            if mode != 0o600 or FeedbackFixture.model_validate_json(raw) != receipt:
                raise ValueError("Rehearsal receipt bytes/mode changed")
            files[relative] = saved[relative] = _digest(raw)
        entries = list(islice(native_dir.iterdir(), 13))
        expected_entries = set(expected_native) | {f"request-{i:02d}.json" for i in range(1, 5)}
        if {p.name for p in entries} != expected_entries:
            raise ValueError("Rehearsal native evidence inventory changed")
        for path in entries:
            raw, mode = _read_regular(path, 2_000_000)
            if mode != 0o600 or _digest(raw) != expected_native.get(path.name, _digest(raw)):
                raise ValueError("Rehearsal native evidence bytes/mode changed")
            name = str(path.relative_to(destination))
            files[name] = saved[name] = _digest(raw)
        rows.append(
            RehearsalCell(cell=cell, workspace_root=str(container / "workspace"), files=files)
        )
        results.append(asdict(result))
    check()
    report_sha = save(
        "report.json",
        json.dumps(
            {"mode": "scripted-real-tools-rehearsal", "results": results}, sort_keys=True
        ).encode(),
    )
    index = FeedbackRehearsal(
        plan_sha256=PLAN_SHA256,
        runtime=runtime,
        settings_sha256=settings_sha,
        command_environment_sha256=env_sha,
        runtime_inventory_sha256=runtime_sha256,
        dependency_files_sha256=dependency_sha256,
        report_sha256=report_sha,
        cells=rows,
    )
    check()
    save("index.json", index.model_dump_json(indent=2).encode())
    return index
