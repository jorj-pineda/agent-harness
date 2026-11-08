"""Production-adapter first inputs for six public cells behind mandatory mock transport."""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Mapping
from itertools import islice
from pathlib import Path
from typing import Annotated, Literal

import httpx
from pydantic import Field

from evals.heldout_command_environment import environment_bytes
from evals.heldout_comparison import ProtocolModel
from evals.heldout_guard import _read_regular
from evals.heldout_inputs import (
    CaptureComplete,
    CaptureProvider,
    _save,
    check_dependency_selection,
    check_runtime_selection,
)
from evals.heldout_settings import (
    SettingsPinnedProvider,
    check_settings,
    settings_bytes,
)
from evals.public_admission import PublicCellWorkspace
from evals.public_envelope import (
    PLAN_SHA256,
    _tree_digest,
    public_settings,
    validate_public_plan,
)
from evals.public_workspaces import _destination, prepare_public_cell
from evals.real_run import _harness_revision, _task_command_environment, load_tasks, run_task
from providers.ollama import OllamaProvider

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class PublicRuntime(ProtocolModel):
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty: Literal[False]


class PublicInputCell(ProtocolModel):
    ordinal: int = Field(ge=1, le=6)
    task_id: str
    repeat: int = Field(ge=1, le=2)
    workspace_root: str
    files: dict[str, Digest] = Field(min_length=5, max_length=5)


class PublicInputIndex(ProtocolModel):
    format: Literal["public-pilot-inputs-v1"] = "public-pilot-inputs-v1"
    plan_sha256: Digest
    runtime: PublicRuntime
    settings_sha256: Digest
    command_environment_sha256: Digest
    runtime_inventory_sha256: Digest
    dependency_files_sha256: Digest
    cells: list[PublicInputCell] = Field(min_length=6, max_length=6)
    generation_forwarded: Literal[0] = 0
    inference_authorized: Literal[False] = False


def _preserve(source: Path, destination: Path, expected: str) -> None:
    raw, mode = _read_regular(source, 2_000_000)
    if mode != 0o600 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Selected public capture input changed before preservation")
    _save(destination, raw)


async def capture_public_inputs(
    directory: Path,
    *,
    command_env: Mapping[str, str],
    runtime_inventory: Path,
    runtime_sha256: str,
    dependency_manifest: Path,
    dependency_sha256: str,
) -> PublicInputIndex:
    plan = validate_public_plan()
    runtime = _harness_revision()
    if runtime["commit"] is None or runtime["dirty"] is not False:
        raise ValueError("Public input capture requires a clean source revision")
    observed_runtime = PublicRuntime.model_validate(runtime)
    selected_env = _task_command_environment(command_env)
    if selected_env is None:
        raise ValueError("Select the public command replacement environment")
    settings = public_settings()
    expected_settings = settings_bytes(settings)
    check_runtime_selection(runtime_inventory, runtime_sha256)
    check_dependency_selection(dependency_manifest, dependency_sha256)
    destination = _destination(directory)
    destination.mkdir(mode=0o700, exist_ok=False)
    settings_sha = _save(destination / "settings.json", expected_settings)
    env_sha = _save(destination / "command-environment.json", environment_bytes(selected_env))
    _preserve(runtime_inventory, destination / "runtime-inventory.json", runtime_sha256)
    _preserve(dependency_manifest, destination / "dependency-files.json", dependency_sha256)
    inventory = destination / "runtime-inventory.json"
    dependencies = destination / "dependency-files.json"
    tasks = {task.id: task for task in load_tasks()}
    rows: list[PublicInputCell] = []

    def check() -> None:
        if (
            destination.is_symlink()
            or _destination(directory) != destination
            or stat.S_IMODE(destination.stat().st_mode) != 0o700
        ):
            raise ValueError("Public capture container changed")
        validate_public_plan()
        check_runtime_selection(inventory, runtime_sha256)
        check_dependency_selection(dependencies, dependency_sha256)
        check_settings(settings, expected_settings)
        for name, digest in (
            ("settings.json", settings_sha),
            ("command-environment.json", env_sha),
        ):
            raw, mode = _read_regular(destination / name, 2_000_000)
            if mode != 0o600 or hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Public Settings/environment capture changed")
        if _harness_revision() != runtime:
            raise ValueError("Public capture runtime changed")

    for cell in plan.cells:
        check()
        container = destination / f"cell-{cell.ordinal:02d}"
        prepare_public_cell(cell.ordinal, container)
        normalized: list[bytes] = []
        wire: list[bytes] = []
        prefix = f"{cell.ordinal:02d}"
        files: dict[str, str] = {}

        def intercepted(
            request: httpx.Request,
            captured: list[bytes] = wire,
            *,
            label: str = prefix,
            normalized_capture: list[bytes] = normalized,
            saved: dict[str, str] = files,
        ) -> httpx.Response:
            if request.method != "POST" or request.url.path != "/api/chat" or captured:
                raise ValueError("Unexpected public first-input request")
            captured.append(request.content)
            saved[f"{label}-wire.json"] = _save(destination / f"{label}-wire.json", request.content)
            if len(normalized_capture) != 1:
                raise ValueError("Public normalized capture unavailable")
            saved[f"{label}-normalized.json"] = _save(
                destination / f"{label}-normalized.json", normalized_capture[0]
            )
            raise CaptureComplete("Mandatory mock: no public request forwarded")

        adapter = OllamaProvider(
            host="http://capture.invalid",
            model=plan.model,
            embed_model="unused",
            timeout_seconds=settings.request_timeout_seconds,
            num_ctx=settings.ollama_num_ctx,
            think=settings.ollama_think,
            transport=httpx.MockTransport(intercepted),
        )
        try:
            result = await run_task(
                tasks[cell.task_id],
                provider=SettingsPinnedProvider(
                    CaptureProvider(adapter, normalized), settings, expected_settings
                ),
                settings=settings,
                mode="harness",
                attempt=cell.repeat,
                public_cell=PublicCellWorkspace(ordinal=cell.ordinal, directory=container),
                command_env=selected_env,
            )
        finally:
            await adapter.aclose()
        check()
        if len(normalized) != 1 or len(wire) != 1 or result.termination != "error:CaptureComplete":
            raise ValueError("Public first-input capture missed its unique mock boundary")
        files[f"{prefix}-runtime-config.json"] = _save(
            destination / f"{prefix}-runtime-config.json",
            json.dumps(result.runtime_config, sort_keys=True).encode(),
        )
        for name in ("prepared.json", "admitted.json"):
            raw, mode = _read_regular(container / name, 4096)
            if mode != 0o600:
                raise ValueError("Public cell receipt mode changed")
            files[f"cell-{cell.ordinal:02d}/{name}"] = hashlib.sha256(raw).hexdigest()
        rows.append(
            PublicInputCell(
                ordinal=cell.ordinal,
                task_id=cell.task_id,
                repeat=cell.repeat,
                workspace_root=str(container / "workspace"),
                files=files,
            )
        )
    check()
    pins = {pin.id: pin for pin in plan.tasks}
    for row in rows:
        for name, digest in row.files.items():
            raw, mode = _read_regular(destination / name, 2_000_000)
            if mode != 0o600 or hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Public first-input evidence changed")
        if _tree_digest(Path(row.workspace_root)) != pins[row.task_id].fixture_sha256:
            raise ValueError("Public mock capture workspace changed")
    expected_outer = {
        "settings.json",
        "command-environment.json",
        "runtime-inventory.json",
        "dependency-files.json",
    }
    for row in rows:
        expected_outer.add(f"cell-{row.ordinal:02d}")
        expected_outer.update(name for name in row.files if "/" not in name)
        container = Path(row.workspace_root).parent
        if {p.name for p in islice(container.iterdir(), 4)} != {
            "workspace",
            "prepared.json",
            "admitted.json",
        }:
            raise ValueError("Public cell capture inventory changed")
    if {p.name for p in islice(destination.iterdir(), len(expected_outer) + 1)} != expected_outer:
        raise ValueError("Public capture inventory changed")
    index = PublicInputIndex(
        plan_sha256=PLAN_SHA256,
        runtime=observed_runtime,
        settings_sha256=settings_sha,
        command_environment_sha256=env_sha,
        runtime_inventory_sha256=runtime_sha256,
        dependency_files_sha256=dependency_sha256,
        cells=rows,
    )
    _save(destination / "index.json", index.model_dump_json(indent=2).encode())
    return index
