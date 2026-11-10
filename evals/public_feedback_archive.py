"""Read-only verification of selected mock evidence, not host revalidation or approval."""

from __future__ import annotations

import json
from pathlib import Path

from evals.heldout_command_environment import EnvironmentPin, load_environment_pin
from evals.heldout_dependency_files import DependencyFiles
from evals.heldout_inputs import normalized_input
from evals.heldout_runtime_guard import RuntimeInventory
from evals.heldout_settings import ProtocolSettings, SettingsPin, load_settings_pin, settings_bytes
from evals.public_envelope import _digest, policy_settings_values
from evals.public_feedback import PLAN_SHA256, validate_feedback_plan, visible_check_bytes
from evals.public_feedback_rehearsal import FeedbackRehearsal
from evals.public_feedback_workspaces import FeedbackFixture, _selection
from evals.public_staging import NormalizedInput, _inventory, _pinned, _private_directory
from evals.public_workspaces import _destination
from providers.ollama import _message_to_ollama, _tool_to_ollama
from workspace.changes import snapshot
from workspace.disposable import COPY_IGNORE_PATTERNS


def verify_feedback_capture(
    directory: Path, *, expected_index_sha256: str, original_directory: Path | None = None
) -> FeedbackRehearsal:
    """Check selected bytes/schemas only; historical executable/dependency paths are not read."""
    source = _destination(directory)
    _private_directory(source)
    index = FeedbackRehearsal.model_validate_json(
        _pinned(source / "index.json", expected_index_sha256)
    )
    plan = validate_feedback_plan()
    if index.plan_sha256 != PLAN_SHA256:
        raise ValueError("Feedback capture plan changed")
    original = source if original_directory is None else original_directory
    if not original.is_absolute() or ".." in original.parts or "\x00" in str(original):
        raise ValueError("Select an absolute historical feedback root")
    settings_raw = load_settings_pin(
        source / "settings.json", expected_sha256=index.settings_sha256
    )
    settings = ProtocolSettings(
        _env_file=None, **SettingsPin.model_validate_json(settings_raw).values
    )
    if settings_bytes(settings) != settings_raw or settings.ollama_model != plan.model:
        raise ValueError("Feedback capture Settings changed")
    values = settings.model_dump()
    policy = policy_settings_values()
    if any(values[key] != value for key, value in policy.items()):
        raise ValueError("Feedback capture runtime policy changed")
    environment = EnvironmentPin.model_validate_json(
        load_environment_pin(source / "command-environment.json", index.command_environment_sha256)
    ).values
    RuntimeInventory.model_validate_json(
        _pinned(source / "runtime-inventory.json", index.runtime_inventory_sha256)
    )
    DependencyFiles.model_validate_json(
        _pinned(source / "dependency-files.json", index.dependency_files_sha256)
    )
    report = json.loads(_pinned(source / "report.json", index.report_sha256))
    if report.get("mode") != "scripted-real-tools-rehearsal" or len(report.get("results", [])) != 8:
        raise ValueError("Feedback rehearsal report identity changed")
    outer = {
        "index.json",
        "settings.json",
        "command-environment.json",
        "runtime-inventory.json",
        "dependency-files.json",
        "report.json",
    }
    for row, cell, result in zip(index.cells, plan.cells, report["results"], strict=True):
        if row.cell != cell:
            raise ValueError("Feedback captured cell order or identity changed")
        prefix = f"{cell.ordinal:02d}"
        container_name, native_name = f"cell-{prefix}", f"native-{prefix}"
        native_files = {
            f"{native_name}/request-{i:02d}{suffix}"
            for i in range(1, 5)
            for suffix in (".request.bin", ".response.bin", ".json")
        }
        names = (
            native_files
            | {
                f"{prefix}-{suffix}"
                for suffix in (
                    "normalized.json",
                    "initial-visible.py",
                    "final-visible.py",
                    "final-implementation.py",
                    "runtime-config.json",
                    "result.json",
                )
            }
            | {f"{container_name}/prepared.json", f"{container_name}/admitted.json"}
        )
        if set(row.files) != names:
            raise ValueError("Feedback captured file map changed")
        expected_root = original / container_name / "workspace"
        if row.workspace_root != str(expected_root):
            raise ValueError("Feedback captured workspace root changed")
        bodies = {name: _pinned(source / name, row.files[name]) for name in names}
        outer.update(Path(name).parts[0] for name in names)
        container, native = source / container_name, source / native_name
        _private_directory(container)
        _private_directory(native)
        _inventory(container, {"workspace", "prepared.json", "admitted.json"})
        _inventory(native, {Path(name).name for name in native_files})
        receipt, task = _selection(cell.ordinal)
        for name in ("prepared.json", "admitted.json"):
            if FeedbackFixture.model_validate_json(bodies[f"{container_name}/{name}"]) != receipt:
                raise ValueError("Feedback captured receipt changed")
        initial = bodies[f"{prefix}-initial-visible.py"]
        if (
            initial != visible_check_bytes(cell.arm)
            or bodies[f"{prefix}-final-visible.py"] != initial
        ):
            raise ValueError("Feedback selected visible test changed")
        workspace = container / "workspace"
        if workspace.is_symlink() or not workspace.is_dir():
            raise ValueError("Invalid archived feedback workspace")
        files = snapshot(
            workspace, ignore=COPY_IGNORE_PATTERNS, max_files=2, max_bytes=1_000_000, strict=True
        )
        if files != {
            "pagination.py": _digest(bodies[f"{prefix}-final-implementation.py"]),
            "test_visible.py": _digest(initial),
        }:
            raise ValueError("Feedback archived final implementation changed")
        norm_raw = bodies[f"{prefix}-normalized.json"]
        normalized = NormalizedInput.model_validate_json(norm_raw)
        if json.loads(norm_raw) != json.loads(
            normalized_input(
                normalized.messages, normalized.tools, normalized.temperature, normalized.max_tokens
            )
        ):
            raise ValueError("Feedback normalized identity changed")
        if (
            [m.role for m in normalized.messages] != ["system", "user"]
            or normalized.messages[1].content != task.request
            or f"Workspace root: {expected_root}" not in normalized.messages[0].content.splitlines()
        ):
            raise ValueError("Feedback first request task or workspace changed")
        offered = {tool.name for tool in normalized.tools}
        if (
            len(offered) != len(normalized.tools)
            or not {"write_file", "run_project_check"} <= offered
            or {"replace_text", "run_command", "emit_plan"} & offered
        ):
            raise ValueError("Feedback first request tool namespace changed")
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
        if json.loads(bodies[f"{native_name}/request-01.request.bin"]) != expected_wire:
            raise ValueError("Feedback native first request differs from normalized input")
        config = json.loads(bodies[f"{prefix}-runtime-config.json"])
        preservation = config.get("public_feedback_preservation", {})
        if (
            config.get("workspace_policy") != "prepared-public-feedback-single-use-v1"
            or config.get("prepared_workspace_root") != str(expected_root)
            or any(config.get(k) != v for k, v in policy.items())
            or config.get("tool_specs") != [t.model_dump(mode="json") for t in normalized.tools]
            or preservation.get("plan_sha256") != PLAN_SHA256
            or preservation.get("ordinal") != cell.ordinal
            or preservation.get("status") != "checks_passed"
            or preservation.get("failed_checkpoint") is not None
        ):
            raise ValueError("Feedback captured runtime configuration changed")
        if config.get("command_environment") != {
            "policy": "replacement-v1",
            "sha256": _digest(json.dumps(environment, sort_keys=True).encode()),
        }:
            raise ValueError("Feedback captured command environment changed")
        if (
            json.loads(bodies[f"{prefix}-result.json"]) != result
            or result.get("runtime_config") != config
            or result.get("task_id") != cell.task_id
            or result.get("attempt") != cell.repeat
            or result.get("mode") != "harness"
            or result.get("passed") is not True
            or result.get("completion_status") != "completed"
            or result.get("verification_status") != "passed"
            or result.get("acceptance_exit_code") != 0
        ):
            raise ValueError("Feedback scripted result differs from report or selected cell")
        prompt_tokens = completion_tokens = 0
        for i in range(1, 5):
            stem = f"{native_name}/request-{i:02d}"
            meta = json.loads(bodies[f"{stem}.json"])
            response = json.loads(bodies[f"{stem}.response.bin"])
            if (
                meta.get("status_code") != 200
                or meta.get("delegated") is not True
                or meta.get("request_bytes") != len(bodies[f"{stem}.request.bin"])
                or meta.get("response_bytes") != len(bodies[f"{stem}.response.bin"])
                or response.get("model") != plan.model
                or response.get("done") is not True
            ):
                raise ValueError("Feedback native response evidence changed")
            if (
                type(response.get("prompt_eval_count")) is not int
                or type(response.get("eval_count")) is not int
            ):
                raise ValueError("Feedback scripted usage unavailable")
            prompt_tokens += response["prompt_eval_count"]
            completion_tokens += response["eval_count"]
        if (result.get("prompt_tokens"), result.get("completion_tokens")) != (
            prompt_tokens,
            completion_tokens,
        ):
            raise ValueError("Feedback scripted usage differs from native evidence")
    _inventory(source, outer)
    _pinned(source / "index.json", expected_index_sha256)
    return index
