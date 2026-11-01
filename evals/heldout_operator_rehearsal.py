"""Named-workspace operator rehearsal with mandatory mock native Ollama transport."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping
from itertools import islice
from pathlib import Path

import httpx

from evals.heldout_command_environment import environment_bytes, load_environment_pin
from evals.heldout_comparison import validate_plan
from evals.heldout_guard import REGISTRATIONS, _read_regular
from evals.heldout_input_guard import PinnedOllamaProvider, load_pins
from evals.heldout_inputs import _save, check_runtime_selection
from evals.heldout_rehearsal import (
    CannedProvider,
    CheckedProvider,
    Fault,
    ProtocolStop,
    stop_reason,
)
from evals.heldout_settings import (
    SettingsPinnedProvider,
    check_settings,
    load_settings_pin,
    protocol_settings,
)
from evals.heldout_staging import verify_capture
from evals.heldout_workspaces import _owned_location, verify_prepared
from evals.real_run import (
    MODES,
    CohortLedger,
    RealTaskResult,
    _harness_revision,
    _plan_cohort,
    _task_command_environment,
    _validate_report_path,
    load_tasks,
    run_task,
    validate_task,
    write_report,
)


def _save_evidence(path: Path, raw: bytes) -> None:
    manifest = REGISTRATIONS[0].root / "tasks.yaml"
    _validate_report_path(path, manifest, load_tasks(manifest))
    _save(path, raw)


class ScriptedTransport(httpx.MockTransport):
    """In-memory responses only; no network transport or injectable delegate."""

    def __init__(self, oracle: CannedProvider, directory: Path) -> None:
        self.oracle, self.directory = oracle, directory
        self.requests = 0
        self.closed = False
        super().__init__(self.reply)

    async def reply(self, request: httpx.Request) -> httpx.Response:
        if self.closed or request.method != "POST" or request.url.path != "/api/chat":
            raise ProtocolStop("Unexpected mock operator route or closed transport")
        self.requests += 1
        _save_evidence(self.directory / f"request-{self.requests:02d}.json", request.content)
        # Deliberate simulation override: the scripted oracle selects reference
        # edits/answers, not a model. Responses traverse the production decoder.
        response = await self.oracle.chat([])
        payload = {
            "model": response.model,
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": response.usage.prompt_tokens,
            "eval_count": response.usage.completion_tokens,
            "message": {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {"function": {"name": call.name, "arguments": call.arguments}}
                    for call in response.tool_calls
                ],
            },
        }
        _save_evidence(
            self.directory / f"response-{self.requests:02d}.json",
            json.dumps(payload, indent=2).encode(),
        )
        return httpx.Response(200, json=payload)

    async def aclose(self) -> None:
        self.closed = True
        await self.oracle.aclose()
        await super().aclose()


def _runtime_matches(expected: dict) -> None:
    if _harness_revision() != expected or expected.get("dirty") is not False:
        raise ValueError("Operator runtime differs from clean capture revision")


def _staged_inventory(staged: Path, capture: Path, digest: str, cells) -> None:
    if staged.is_symlink() or {p.name for p in islice(staged.iterdir(), 18)} != {
        "staged.json",
        *(f"cell-{cell.ordinal:02d}" for cell in cells),
    }:
        raise ValueError("Staged root inventory changed")
    raw, mode = _read_regular(staged / "staged.json", 4096)
    if mode != 0o600 or json.loads(raw) != {
        "format": "staged-heldout-capture-v1",
        "capture_index_sha256": digest,
        "archive": str(capture),
        "inference_authorized": False,
    }:
        raise ValueError("Staging receipt differs from selected capture")


async def rehearse(
    staged_directory: Path,
    capture_directory: Path,
    directory: Path,
    *,
    expected_index_sha256: str,
    fault: Fault = "none",
    command_env: Mapping[str, str] | None = None,
    runtime_sha256: str | None = None,
) -> CohortLedger:
    selected_env = _task_command_environment(command_env)
    env_kwargs = {"command_env": selected_env} if selected_env is not None else {}
    plan = validate_plan()
    staged, capture, destination = (
        _owned_location(path) for path in (staged_directory, capture_directory, directory)
    )
    for protected in (staged, capture):
        if destination.is_relative_to(protected) or protected.is_relative_to(destination):
            raise ValueError("Operator output overlaps staged/captured artifacts")
    index = verify_capture(
        capture, expected_index_sha256=expected_index_sha256, original_directory=staged
    )
    runtime = index.runtime.model_dump()
    _runtime_matches(runtime)
    if index.settings_sha256 is None:
        raise ValueError("Capture lacks validated settings pin; legacy capture is not admissible")
    expected_settings = load_settings_pin(
        capture / "settings.json", expected_sha256=index.settings_sha256
    )
    if index.command_environment_sha256 is not None:
        expected_environment = load_environment_pin(
            capture / "command-environment.json", index.command_environment_sha256
        )
        if selected_env is None or environment_bytes(selected_env) != expected_environment:
            raise ValueError("Selected command environment differs from capture")
    elif selected_env is not None:
        raise ValueError("Capture lacks selected command environment pin")
    if runtime_sha256 != index.runtime_inventory_sha256:
        raise ValueError("Selected runtime hash differs from capture")
    runtime_path = capture / "runtime-inventory.json" if runtime_sha256 is not None else None
    check_runtime_selection(runtime_path, runtime_sha256)
    settings = protocol_settings(plan)
    check_settings(settings, expected_settings)
    _staged_inventory(staged, capture, expected_index_sha256, index.cells)
    for cell in index.cells:
        verify_prepared(staged / f"cell-{cell.ordinal:02d}", task_id=cell.task_id)
    manifest = REGISTRATIONS[0].root / "tasks.yaml"
    tasks = load_tasks(manifest)
    ledger, schedule = _plan_cohort(tasks, MODES, plan.repeats)
    destination.mkdir(mode=0o700, exist_ok=False)
    results: list[RealTaskResult] = []

    def persist() -> None:
        _validate_report_path(destination / "report.json", manifest, tasks)
        write_report(
            destination / "report.json",
            "scripted named-workspace native-adapter rehearsal; no inference",
            results,
            cohort=ledger,
            scripted=True,
        )

    persist()
    try:
        for task in tasks:
            check_runtime_selection(runtime_path, runtime_sha256)
            await validate_task(task, **env_kwargs)
        ledger.status = "running"
        for ordinal, (task, mode, attempt) in enumerate(schedule, 1):
            _runtime_matches(runtime)
            check_runtime_selection(runtime_path, runtime_sha256)
            check_settings(settings, expected_settings)
            _staged_inventory(staged, capture, expected_index_sha256, index.cells)
            cell = ledger.cells[ordinal - 1]
            cell.status = "started"
            persist()
            pins = load_pins(
                capture,
                original_directory=staged,
                expected_index_sha256=expected_index_sha256,
                ordinal=ordinal,
            )
            cell_output = destination / f"cell-{ordinal:02d}"
            cell_output.mkdir(mode=0o700)
            transport = ScriptedTransport(
                CannedProvider(task, plan, fault if ordinal == 1 else "none"), cell_output
            )
            provider = None
            failure = None
            try:
                provider = CheckedProvider(
                    SettingsPinnedProvider(
                        PinnedOllamaProvider(
                            pins=pins,
                            host="http://127.0.0.1:11447",
                            model=plan.model,
                            num_ctx=plan.runtime.ollama_num_ctx,
                            think=plan.runtime.ollama_think,
                            timeout_seconds=plan.runtime.request_timeout_seconds,
                            transport=transport,
                        ),
                        settings,
                        expected_settings,
                    ),
                    plan,
                )
                result = await run_task(
                    task,
                    provider=provider,
                    settings=settings,
                    mode=mode,
                    attempt=attempt,
                    prepared_directory=staged / f"cell-{ordinal:02d}",
                    **env_kwargs,
                )
                cell.result_index = len(results)
                results.append(result)
                cell.status = "recorded"
                persist()
                if result.runtime_config != index.cells[ordinal - 1].runtime_config:
                    raise ProtocolStop("runtime_configuration")
                reason = provider.stop_reason or stop_reason(result, plan)
                if reason:
                    raise ProtocolStop(reason)
            except BaseException as error:
                failure = error
                raise
            finally:
                try:
                    if provider is not None:
                        await provider.aclose()
                    else:
                        await transport.aclose()
                except BaseException as error:
                    ledger.cleanup_error = type(error).__name__
                    if failure is None:
                        raise
        _runtime_matches(runtime)
        check_settings(settings, expected_settings)
        verify_capture(
            capture, expected_index_sha256=expected_index_sha256, original_directory=staged
        )
        check_runtime_selection(runtime_path, runtime_sha256)
        ledger.status = "finished"
    except BaseException as error:
        ledger.stop(str(error) if isinstance(error, ProtocolStop) else type(error).__name__)
        raise
    finally:
        persist()
        _save_evidence(
            destination / "operator.json",
            json.dumps(
                {
                    "format": "heldout-operator-rehearsal-v1",
                    "evidence_kind": "scripted-native-adapter-real-tools",
                    "runtime": runtime,
                    "capture_index_sha256": expected_index_sha256,
                    "settings_sha256": index.settings_sha256,
                    "command_environment_sha256": index.command_environment_sha256,
                    "runtime_inventory_sha256": index.runtime_inventory_sha256,
                    "status": ledger.status,
                    "stop_reason": ledger.stop_reason,
                    "cleanup_error": ledger.cleanup_error,
                    "forwarded_requests": 0,
                    "inference_ran": False,
                    "usage_kind": "scripted",
                    "inference_authorized": False,
                },
                indent=2,
            ).encode(),
        )
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged-directory", type=Path, required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument(
        "--command-path", help="Explicit replacement PATH; locale C, no ambient merge"
    )
    parser.add_argument(
        "--fault", choices=("none", "missing_usage", "wrong_model", "transport"), default="none"
    )
    parser.add_argument("--runtime-sha256")
    args = parser.parse_args()
    asyncio.run(
        rehearse(
            args.staged_directory,
            args.capture_directory,
            args.directory,
            expected_index_sha256=args.index_sha256,
            fault=args.fault,
            runtime_sha256=args.runtime_sha256,
            command_env={"PATH": args.command_path, "LC_ALL": "C"}
            if args.command_path is not None
            else None,
        )
    )
    print("Scripted operator rehearsal finished; zero requests forwarded, no launch approval.")


if __name__ == "__main__":
    main()
