"""Mandatory-mock fresh-task native capture through shared coding evaluation."""

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

from evals.fresh_visible_workspaces import (
    SELECTION_SHA256,
    FreshCell,
    FreshCellWorkspace,
    FreshFixture,
    prepare_fresh_cell,
    selected_cell,
)
from evals.heldout_command_environment import environment_bytes
from evals.heldout_comparison import ProtocolModel
from evals.heldout_guard import _read_regular
from evals.heldout_inputs import _save, check_dependency_selection, check_runtime_selection
from evals.heldout_native_evidence import NativeEvidenceTransport
from evals.heldout_settings import check_settings, settings_bytes
from evals.public_envelope import _digest, public_settings
from evals.public_feedback_rehearsal import _CapturedProvider
from evals.public_inputs import Digest, PublicRuntime
from evals.public_workspaces import _destination
from evals.real_run import _harness_revision, _task_command_environment, run_task
from providers.ollama import OllamaProvider

MODULES = {
    "fv_lookup_option": "lookup.py",
    "fv_merge_windows": "windows.py",
    "fv_header_pair": "headers.py",
}


class CapturedFreshCell(ProtocolModel):
    cell: FreshCell
    workspace_root: str
    files: dict[str, Digest] = Field(min_length=17, max_length=17)


class FreshRehearsal(ProtocolModel):
    format: Literal["fresh-visible-mock-rehearsal-v1"] = "fresh-visible-mock-rehearsal-v1"
    selection_sha256: Digest
    runtime: PublicRuntime
    settings_sha256: Digest
    command_environment_sha256: Digest
    runtime_inventory_sha256: Digest
    dependency_files_sha256: Digest
    report_sha256: Digest
    cells: list[CapturedFreshCell] = Field(min_length=12, max_length=12)
    mock_generation_requests: Literal[36] = 36
    generation_forwarded: Literal[0] = 0
    inference_authorized: Literal[False] = False
    evidence_label: Literal["scripted-real-tools-rehearsal"] = "scripted-real-tools-rehearsal"


class _ReferenceTransport(httpx.MockTransport):
    """Fixed write/check/final native responses. No network delegate or live option."""

    def __init__(self, model: str, module: str, reference: str) -> None:
        self.model, self.module, self.reference = model, module, reference
        self.requests = 0
        self.closed = False
        self.native_pins: dict[str, str] = {}
        super().__init__(self.reply)

    def reply(self, request: httpx.Request) -> httpx.Response:
        if (
            self.closed
            or request.method != "POST"
            or str(request.url) != "http://127.0.0.1:11447/api/chat"
            or self.requests >= 3
        ):
            raise ValueError("Unexpected fresh rehearsal mock request")
        payload = json.loads(request.content)
        if payload["model"] != self.model or not {"write_file", "run_project_check"} <= {
            tool["function"]["name"] for tool in payload["tools"]
        }:
            raise ValueError("Fresh rehearsal model/tools differ")
        self.requests += 1
        if self.requests == 1:
            calls = [("write_file", {"path": self.module, "content": self.reference})]
        elif self.requests == 2:
            calls = [("run_project_check", {})]
        else:
            if json.loads(payload["messages"][-1]["content"])["exit_code"] != 0:
                raise ValueError("Fresh rehearsal reference check did not pass")
            calls = []
        response = httpx.Response(
            200,
            json={
                "model": self.model,
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 5,
                "message": {
                    "role": "assistant",
                    "content": "Scripted completion" if not calls else "",
                    "tool_calls": [
                        {"function": {"name": name, "arguments": args}} for name, args in calls
                    ],
                },
            },
        )
        for suffix, raw in (("request.bin", request.content), ("response.bin", response.content)):
            self.native_pins[f"request-{self.requests:02d}.{suffix}"] = _digest(raw)
        return response

    async def aclose(self) -> None:
        self.closed = True
        await super().aclose()


class _FreshCapture:
    """Private serial capture driver shared by offline and owned-mock entry points."""

    def __init__(
        self,
        directory: Path,
        *,
        command_env: Mapping[str, str],
        runtime_inventory: Path,
        runtime_sha256: str,
        dependency_manifest: Path,
        dependency_sha256: str,
    ) -> None:
        self.cells = [selected_cell(ordinal)[0] for ordinal in range(1, 13)]
        self.observed = _harness_revision()
        self.runtime = PublicRuntime.model_validate(self.observed)
        self.environment = _task_command_environment(command_env)
        if self.environment is None:
            raise ValueError("Select a replacement command environment")
        self.settings = public_settings()
        self.expected_settings = settings_bytes(self.settings)
        self.runtime_inventory, self.runtime_sha256 = runtime_inventory, runtime_sha256
        self.dependency_manifest, self.dependency_sha256 = dependency_manifest, dependency_sha256
        check_runtime_selection(runtime_inventory, runtime_sha256)
        check_dependency_selection(dependency_manifest, dependency_sha256)
        self.directory = directory
        self.destination = _destination(directory)
        self.destination.mkdir(mode=0o700, exist_ok=False)
        self.saved: dict[str, str] = {}
        self.outer: set[str] = set()
        self.rows: list[CapturedFreshCell] = []
        self.results: list[dict] = []
        try:
            self.settings_sha = self.save("settings.json", self.expected_settings)
            self.environment_sha = self.save(
                "command-environment.json", environment_bytes(self.environment)
            )
            for source, expected, name in (
                (runtime_inventory, runtime_sha256, "runtime-inventory.json"),
                (dependency_manifest, dependency_sha256, "dependency-files.json"),
            ):
                raw, mode = _read_regular(source, 2_000_000)
                if mode != 0o600 or _digest(raw) != expected:
                    raise ValueError("Fresh rehearsal selected host inventory changed")
                self.save(name, raw)
        except BaseException as error:
            self.fail(error)
            raise

    def save(self, name: str, raw: bytes) -> str:
        digest = _save(self.destination / name, raw)
        self.saved[name] = digest
        self.outer.add(Path(name).parts[0])
        return digest

    def retain(self, name: str, expected: str | None = None) -> str:
        raw, mode = _read_regular(self.destination / name, 2_000_000)
        if mode != 0o600 or (expected is not None and _digest(raw) != expected):
            raise ValueError("Fresh rehearsal evidence bytes/mode changed")
        self.saved[name] = _digest(raw)
        return self.saved[name]

    def check(self) -> None:
        if (
            self.destination.is_symlink()
            or _destination(self.directory) != self.destination
            or stat.S_IMODE(self.destination.stat().st_mode) != 0o700
            or _harness_revision() != self.observed
        ):
            raise ValueError("Fresh rehearsal container/runtime changed")
        check_settings(self.settings, self.expected_settings)
        check_runtime_selection(self.runtime_inventory, self.runtime_sha256)
        check_dependency_selection(self.dependency_manifest, self.dependency_sha256)
        for name, expected in self.saved.items():
            raw, mode = _read_regular(self.destination / name, 2_000_000)
            if mode != 0o600 or _digest(raw) != expected:
                raise ValueError("Fresh rehearsal preserved evidence changed")
        if {p.name for p in islice(self.destination.iterdir(), len(self.outer) + 1)} != self.outer:
            raise ValueError("Fresh rehearsal inventory changed")

    async def capture_cell(
        self, cell: FreshCell, *, check_owned: Callable[[], None] | None = None
    ) -> None:
        if len(self.rows) >= 12 or cell != self.cells[len(self.rows)]:
            raise ValueError("Fresh capture cell differs from fixed serial prefix")

        def check() -> None:
            self.check()
            if check_owned is not None:
                check_owned()

        destination, settings, environment = self.destination, self.settings, self.environment
        save, retain, outer, rows, results = (
            self.save,
            self.retain,
            self.outer,
            self.rows,
            self.results,
        )
        check()
        _, task, _, _ = selected_cell(cell.ordinal)
        prefix = f"{cell.ordinal:02d}"
        container = destination / f"cell-{prefix}"
        receipt = prepare_fresh_cell(cell.ordinal, container)
        outer.add(container.name)
        files: dict[str, str] = {}
        initial, _ = _read_regular(container / "workspace/test_visible.py", 16_000)
        files[f"{prefix}-initial-visible.py"] = save(f"{prefix}-initial-visible.py", initial)
        module = MODULES[cell.task_id]
        reference, _ = _read_regular(task.reference / module, 64_000)
        transport = _ReferenceTransport(settings.ollama_model, module, reference.decode())
        native_dir = destination / f"native-{prefix}"
        outer.add(native_dir.name)
        native = NativeEvidenceTransport(transport, native_dir)
        adapter = OllamaProvider(
            host=settings.ollama_host,
            model=settings.ollama_model,
            embed_model="unused",
            timeout_seconds=settings.request_timeout_seconds,
            num_ctx=settings.ollama_num_ctx,
            think=settings.ollama_think,
            transport=native,
        )

        def capture(raw: bytes, label: str = prefix, pins: dict[str, str] = files) -> None:
            name = f"{label}-normalized.json"
            pins[name] = save(name, raw)

        primary = None
        try:
            result = await run_task(
                task,
                provider=_CapturedProvider(adapter, check, capture),
                settings=settings,
                mode="harness",
                attempt=cell.repeat,
                fresh_cell=FreshCellWorkspace(ordinal=cell.ordinal, directory=container),
                command_env=environment,
            )
            native_failed = native.failed
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                await adapter.aclose()
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note(
                    f"Fresh rehearsal transport cleanup also failed: {type(cleanup_error).__name__}"
                )
        name = f"{prefix}-result.json"
        files[name] = save(name, json.dumps(asdict(result), sort_keys=True).encode())
        check()
        if (
            transport.requests != 3
            or not transport.closed
            or native.requests != 3
            or native_failed
            or not native.closed
            or not result.passed
            or result.completion_status != "completed"
            or result.verification_status != "passed"
            or result.runtime_config["fresh_visible_preservation"]["status"] != "checks_passed"
        ):
            raise ValueError("Fresh mandatory-mock reference rehearsal incomplete")
        final, _ = _read_regular(container / "workspace/test_visible.py", 16_000)
        implementation, _ = _read_regular(container / "workspace" / module, 64_000)
        if final != initial or implementation != reference or receipt.cell != cell:
            raise ValueError("Fresh rehearsal final bytes/identity differ")
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
            raise ValueError("Fresh rehearsal cell inventory changed")
        for label in ("prepared.json", "admitted.json"):
            name = f"{container.name}/{label}"
            if (
                FreshFixture.model_validate_json(_read_regular(destination / name, 4096)[0])
                != receipt
            ):
                raise ValueError("Fresh rehearsal cell receipt changed")
            files[name] = retain(name)
        names = set(transport.native_pins) | {f"request-{n:02d}.json" for n in range(1, 4)}
        if {p.name for p in islice(native_dir.iterdir(), 10)} != names:
            raise ValueError("Fresh native evidence inventory changed")
        for name in sorted(names):
            relative = str((native_dir / name).relative_to(destination))
            files[relative] = retain(relative, transport.native_pins.get(name))
        rows.append(
            CapturedFreshCell(cell=cell, workspace_root=str(container / "workspace"), files=files)
        )
        results.append(asdict(result))

    def finish(self) -> FreshRehearsal:
        if len(self.rows) != 12:
            raise ValueError("Fresh capture requires all twelve cells before success")
        check, save, results, rows = self.check, self.save, self.results, self.rows
        runtime = self.runtime
        settings_sha, environment_sha = self.settings_sha, self.environment_sha
        runtime_sha256, dependency_sha256 = self.runtime_sha256, self.dependency_sha256
        check()
        report_sha = save(
            "report.json",
            json.dumps(
                {"mode": "scripted-real-tools-rehearsal", "results": results}, sort_keys=True
            ).encode(),
        )
        summary = FreshRehearsal(
            selection_sha256=SELECTION_SHA256,
            runtime=runtime,
            settings_sha256=settings_sha,
            command_environment_sha256=environment_sha,
            runtime_inventory_sha256=runtime_sha256,
            dependency_files_sha256=dependency_sha256,
            report_sha256=report_sha,
            cells=rows,
        )
        check()
        save("index.json", summary.model_dump_json(indent=2).encode())
        return summary

    def fail(self, error: BaseException) -> None:
        try:
            _save(
                self.destination / "failure.json",
                json.dumps(
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "inference_authorized": False,
                        "retry_authorized": False,
                    }
                ).encode(),
            )
        except BaseException as secondary:
            error.add_note(
                f"Fresh rehearsal failure recording also failed: {type(secondary).__name__}"
            )


async def rehearse_fresh_inputs(
    directory: Path,
    *,
    command_env: Mapping[str, str],
    runtime_inventory: Path,
    runtime_sha256: str,
    dependency_manifest: Path,
    dependency_sha256: str,
) -> FreshRehearsal:
    """Consume twelve newly prepared cells, save evidence privately, publish index last."""
    capture = _FreshCapture(
        directory,
        command_env=command_env,
        runtime_inventory=runtime_inventory,
        runtime_sha256=runtime_sha256,
        dependency_manifest=dependency_manifest,
        dependency_sha256=dependency_sha256,
    )
    try:
        for cell in capture.cells:
            await capture.capture_cell(cell)
        return capture.finish()
    except BaseException as error:
        capture.fail(error)
        raise
