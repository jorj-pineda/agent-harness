"""Verify and preserve public first inputs before fresh exact-path cell staging."""

from __future__ import annotations

import hashlib
import json
import stat
from itertools import islice
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from evals.heldout_command_environment import EnvironmentPin, load_environment_pin
from evals.heldout_comparison import ProtocolModel
from evals.heldout_dependency_files import DependencyFiles
from evals.heldout_guard import _read_regular
from evals.heldout_inputs import _save, normalized_input
from evals.heldout_runtime_guard import RuntimeInventory
from evals.heldout_settings import ProtocolSettings, SettingsPin, load_settings_pin, settings_bytes
from evals.public_envelope import (
    PLAN_SHA256,
    _tree_digest,
    policy_settings_values,
    validate_public_plan,
)
from evals.public_inputs import Digest, PublicInputIndex
from evals.public_workspaces import (
    PublicFixture,
    _destination,
    prepare_public_cell,
    verify_public_cell,
)
from evals.real_run import load_tasks
from harness.trace import RequestIdentity
from providers.base import ChatMessage, ToolSpec
from providers.ollama import _message_to_ollama, _tool_to_ollama


class NormalizedInput(ProtocolModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)
    identity: RequestIdentity
    messages: list[ChatMessage] = Field(min_length=2, max_length=2)
    tools: list[ToolSpec] = Field(min_length=1, max_length=64)
    temperature: float = Field(ge=0, le=0, allow_inf_nan=False)
    max_tokens: Literal[2048]


class StagedPublic(ProtocolModel):
    format: Literal["staged-public-pilot-v1"] = "staged-public-pilot-v1"
    plan_sha256: Digest
    capture_index_sha256: Digest
    archive: str
    inference_authorized: Literal[False] = False


def _private_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir() or stat.S_IMODE(path.stat().st_mode) != 0o700:
        raise ValueError("Public artifact container type or mode changed")


def _inventory(path: Path, expected: set[str]) -> None:
    # Unique names: one beyond the expected cardinality detects every stationary extra.
    if {p.name for p in islice(path.iterdir(), len(expected) + 1)} != expected:
        raise ValueError("Public artifact inventory changed")


def _pinned(path: Path, digest: str, limit: int = 2_000_000) -> bytes:
    raw, mode = _read_regular(path, limit)
    if mode != 0o600 or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Public captured file bytes or private mode changed")
    return raw


def verify_public_capture(
    directory: Path, *, expected_index_sha256: str, original_directory: Path | None = None
) -> PublicInputIndex:
    source = _destination(directory)
    _private_directory(source)
    raw = _pinned(source / "index.json", expected_index_sha256)
    try:
        index = PublicInputIndex.model_validate_json(raw)
    except ValueError:
        raise ValueError("Invalid public capture index schema") from None
    plan = validate_public_plan()
    if index.plan_sha256 != PLAN_SHA256:
        raise ValueError("Public capture plan changed")
    original = original_directory if original_directory is not None else source
    if not original.is_absolute() or ".." in original.parts or "\x00" in str(original):
        raise ValueError("Select an absolute historical capture root")
    settings_raw = load_settings_pin(
        source / "settings.json", expected_sha256=index.settings_sha256
    )
    settings = ProtocolSettings(
        _env_file=None, **SettingsPin.model_validate_json(settings_raw).values
    )
    if settings_bytes(settings) != settings_raw or settings.ollama_model != plan.model:
        raise ValueError("Public capture Settings are not canonical selected settings")
    values = settings.model_dump()
    if any(values[key] != value for key, value in policy_settings_values().items()):
        raise ValueError("Public capture policy changed")
    env_raw = load_environment_pin(
        source / "command-environment.json", index.command_environment_sha256
    )
    environment = EnvironmentPin.model_validate_json(env_raw).values
    RuntimeInventory.model_validate_json(
        _pinned(source / "runtime-inventory.json", index.runtime_inventory_sha256)
    )
    DependencyFiles.model_validate_json(
        _pinned(source / "dependency-files.json", index.dependency_files_sha256)
    )
    # Archive checks read schemas/bytes, not historical executable/dependency paths.
    tasks = {task.id: task for task in load_tasks()}
    pins = {pin.id: pin for pin in plan.tasks}
    outer = {
        "index.json",
        "settings.json",
        "command-environment.json",
        "runtime-inventory.json",
        "dependency-files.json",
    }
    for row, cell in zip(index.cells, plan.cells, strict=True):
        if (row.ordinal, row.task_id, row.repeat) != (cell.ordinal, cell.task_id, cell.repeat):
            raise ValueError("Public capture cell order changed")
        prefix = f"{cell.ordinal:02d}"
        container_name = f"cell-{prefix}"
        names = {
            f"{prefix}-normalized.json",
            f"{prefix}-wire.json",
            f"{prefix}-runtime-config.json",
            f"{container_name}/prepared.json",
            f"{container_name}/admitted.json",
        }
        if set(row.files) != names:
            raise ValueError("Public capture file map changed")
        expected_root = original / container_name / "workspace"
        if row.workspace_root != str(expected_root):
            raise ValueError("Public capture original workspace root changed")
        bodies = {name: _pinned(source / name, row.files[name]) for name in names}
        container = source / container_name
        _private_directory(container)
        _inventory(container, {"workspace", "prepared.json", "admitted.json"})
        receipt = PublicFixture(
            plan_sha256=PLAN_SHA256, ordinal=cell.ordinal, task_id=cell.task_id, repeat=cell.repeat
        )
        if any(
            PublicFixture.model_validate_json(bodies[f"{container_name}/{name}"]) != receipt
            for name in ("prepared.json", "admitted.json")
        ):
            raise ValueError("Public captured cell receipt changed")
        if _tree_digest(container / "workspace") != pins[cell.task_id].fixture_sha256:
            raise ValueError("Public captured workspace changed")
        norm_raw = bodies[f"{prefix}-normalized.json"]
        normalized = NormalizedInput.model_validate_json(norm_raw)
        if json.loads(norm_raw) != json.loads(
            normalized_input(
                normalized.messages, normalized.tools, normalized.temperature, normalized.max_tokens
            )
        ):
            raise ValueError("Public normalized request identity changed")
        if [m.role for m in normalized.messages] != ["system", "user"] or normalized.messages[
            1
        ].content != tasks[cell.task_id].request:
            raise ValueError("Public first request differs from selected task")
        if f"Workspace root: {expected_root}" not in normalized.messages[0].content.splitlines():
            raise ValueError("Public first request has a different workspace root")
        names_offered = {tool.name for tool in normalized.tools}
        if (
            len(names_offered) != len(normalized.tools)
            or not {"write_file", "run_project_check"} <= names_offered
            or {"replace_text", "run_command", "emit_plan"} & names_offered
        ):
            raise ValueError("Public capture tool namespace conflicts with selected policy")
        expected_wire = {
            "model": plan.model,
            "messages": [_message_to_ollama(m) for m in normalized.messages],
            "stream": False,
            "options": {
                "temperature": normalized.temperature,
                "num_predict": normalized.max_tokens,
                "num_ctx": settings.ollama_num_ctx,
            },
            "think": settings.ollama_think,
            "tools": [_tool_to_ollama(t) for t in normalized.tools],
        }
        if json.loads(bodies[f"{prefix}-wire.json"]) != expected_wire:
            raise ValueError("Public native request differs from normalized input/settings")
        config = json.loads(bodies[f"{prefix}-runtime-config.json"])
        if (
            config.get("workspace_policy") != "prepared-public-pilot-single-use-v1"
            or config.get("prepared_workspace_root") != str(expected_root)
            or any(config.get(key) != value for key, value in policy_settings_values().items())
            or config.get("tool_specs") != [t.model_dump(mode="json") for t in normalized.tools]
        ):
            raise ValueError("Public captured runtime configuration changed")
        preservation = config.get("public_pilot_preservation", {})
        if (
            preservation.get("plan_sha256") != PLAN_SHA256
            or preservation.get("ordinal") != cell.ordinal
            or preservation.get("status") != "checks_passed"
            or preservation.get("failed_checkpoint") is not None
        ):
            raise ValueError("Public capture preservation did not pass")
        env_digest = hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()
        if config.get("command_environment") != {"policy": "replacement-v1", "sha256": env_digest}:
            raise ValueError("Public captured command environment changed")
        outer.add(container_name)
        outer.update(name for name in names if "/" not in name)
    _inventory(source, outer)
    _pinned(source / "index.json", expected_index_sha256)
    return index


def stage_public_capture(directory: Path, archive: Path, *, expected_index_sha256: str) -> None:
    source, target = _destination(directory), _destination(archive)
    if source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Public capture and archive paths overlap")
    index = verify_public_capture(source, expected_index_sha256=expected_index_sha256)
    target.mkdir(mode=0o700, exist_ok=False)
    _inventory(target, set())
    source.rename(target / "captured")
    verify_public_capture(
        target / "captured", expected_index_sha256=expected_index_sha256, original_directory=source
    )
    _inventory(target, {"captured"})
    source.mkdir(mode=0o700, exist_ok=False)
    for cell in index.cells:
        container = source / f"cell-{cell.ordinal:02d}"
        prepare_public_cell(cell.ordinal, container)
        verify_public_cell(container, ordinal=cell.ordinal)
    _inventory(source, {f"cell-{cell.ordinal:02d}" for cell in index.cells})
    verify_public_capture(
        target / "captured", expected_index_sha256=expected_index_sha256, original_directory=source
    )
    for cell in index.cells:
        verify_public_cell(source / f"cell-{cell.ordinal:02d}", ordinal=cell.ordinal)
    _private_directory(target)
    _inventory(target, {"captured"})
    _save(
        source / "staged.json",
        StagedPublic(
            plan_sha256=PLAN_SHA256,
            capture_index_sha256=expected_index_sha256,
            archive=str(target / "captured"),
        )
        .model_dump_json(indent=2)
        .encode(),
    )


def verify_staged_public(
    directory: Path, archive: Path, *, expected_index_sha256: str
) -> PublicInputIndex:
    source, target = _destination(directory), _destination(archive)
    if source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Public staged and archive paths overlap")
    _private_directory(source)
    _private_directory(target)
    _inventory(target, {"captured"})
    raw, mode = _read_regular(source / "staged.json", 4096)
    expected = StagedPublic(
        plan_sha256=PLAN_SHA256,
        capture_index_sha256=expected_index_sha256,
        archive=str(target / "captured"),
    )
    if mode != 0o600 or StagedPublic.model_validate_json(raw) != expected:
        raise ValueError("Public staged receipt changed")
    index = verify_public_capture(
        target / "captured", expected_index_sha256=expected_index_sha256, original_directory=source
    )
    _inventory(source, {"staged.json", *(f"cell-{cell.ordinal:02d}" for cell in index.cells)})
    for cell in index.cells:
        verify_public_cell(source / f"cell-{cell.ordinal:02d}", ordinal=cell.ordinal)
    return index
