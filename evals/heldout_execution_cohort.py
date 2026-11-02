"""Execute a fixed admitted cohort with per-cell owned settlement; no launch CLI."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from pathlib import Path

import httpx

from evals.heldout_backend_rehearsal import owned_backend
from evals.heldout_comparison import PLAN_PATH, validate_plan
from evals.heldout_execution_admission import ExecutionAdmission
from evals.heldout_execution_manifest import _selected_bytes
from evals.heldout_execution_provider import build_cell_provider
from evals.heldout_guard import REGISTRATIONS
from evals.heldout_inputs import _save, check_dependency_selection, check_runtime_selection
from evals.heldout_model_image import _hash_blob
from evals.heldout_native_evidence import NativeEvidenceTransport
from evals.heldout_operator_rehearsal import _runtime_matches, _staged_inventory
from evals.heldout_rehearsal import ProtocolStop, stop_reason
from evals.heldout_settings import protocol_settings
from evals.heldout_staging import verify_capture
from evals.heldout_workspaces import _owned_location, verify_prepared
from evals.real_run import (
    MODES,
    CohortLedger,
    _plan_cohort,
    _validate_report_path,
    load_tasks,
    run_task,
    validate_task,
    write_report,
)


def run_admitted_cohort(
    admission: ExecutionAdmission,
    *,
    transport_factory: Callable[[int, Path], httpx.AsyncBaseTransport],
    scripted: bool,
) -> CohortLedger:
    """Trusted caller must have completed real admission and launch prerequisites.

    Explicit transport selection has no default. Transport factories must not submit
    requests themselves. Lifecycle entry/exit is synchronous, outside event loops.
    This API cannot authenticate a constructed Admission object or approve a launch.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise ValueError("Run the owned cohort outside an active event loop")
    manifest = admission.manifest.model_copy(deep=True)
    staged, capture = Path(manifest.staged_directory), Path(manifest.capture_directory)
    destination = admission.directory
    plan = validate_plan()
    tasks = load_tasks(REGISTRATIONS[0].root / "tasks.yaml")
    ledger, schedule = _plan_cohort(tasks, MODES, plan.repeats)
    settings = protocol_settings(plan)
    results = []

    def check() -> None:
        _selected_bytes(destination / "manifest.json", admission.approval.manifest_sha256, 64_000)
        _selected_bytes(
            Path(manifest.image_directory) / "prepared-image.json",
            manifest.image_receipt_sha256,
            64_000,
        )
        binary = Path(manifest.ollama_binary)
        size = binary.stat().st_size
        if (
            not 1 <= size <= 256_000_000
            or _hash_blob(binary, size)["sha256"] != manifest.binary_sha256
        ):
            raise ValueError("Selected executable changed")
        _runtime_matches({"commit": manifest.runtime_commit, "dirty": False})
        if hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() != manifest.protocol_sha256:
            raise ValueError("Protocol changed after admission")
        verify_capture(
            capture, expected_index_sha256=manifest.capture_index_sha256, original_directory=staged
        )
        check_runtime_selection(
            capture / "runtime-inventory.json", manifest.runtime_inventory_sha256
        )
        check_dependency_selection(
            capture / "dependency-files.json", manifest.dependency_files_sha256
        )
        _staged_inventory(staged, capture, manifest.capture_index_sha256, admission.capture.cells)

    def persist() -> None:
        if (
            destination.is_symlink()
            or destination.resolve() != destination
            or _owned_location(destination) != destination
        ):
            raise ValueError("Cohort output path changed")
        _validate_report_path(
            destination / "report.json", REGISTRATIONS[0].root / "tasks.yaml", tasks
        )
        write_report(
            destination / "report.json",
            "admitted held-out fixed cohort",
            results,
            cohort=ledger,
            scripted=scripted,
        )

    async def controls() -> None:
        for task in tasks:
            check()
            await validate_task(task, command_env=manifest.command_environment)

    check()
    for pinned in admission.capture.cells:
        verify_prepared(staged / f"cell-{pinned.ordinal:02d}", task_id=pinned.task_id)
    # Exclusive marker prevents re-entry even after controls or first report fail.
    _save(destination / "cohort-started.json", b'{"format":"heldout-cohort-started-v1"}')
    try:
        persist()
        asyncio.run(controls())
        ledger.status = "running"
        for ordinal, (task, mode, attempt) in enumerate(schedule, 1):
            check()
            cell = ledger.cells[ordinal - 1]
            cell.status = "started"
            persist()
            cell_output = destination / f"cell-{ordinal:02d}"
            cell_output.mkdir(mode=0o700)

            async def execute(
                check_owned,
                ordinal=ordinal,
                cell_output=cell_output,
                task=task,
                mode=mode,
                attempt=attempt,
                cell=cell,
            ):
                transport = transport_factory(ordinal, cell_output)
                provider = None
                primary = None
                try:
                    transport = NativeEvidenceTransport(transport, cell_output / "native")
                    provider = build_cell_provider(
                        admission,
                        ordinal,
                        settings=settings,
                        transport=transport,
                        check_owned=check_owned,
                    )
                    result = await run_task(
                        task,
                        provider=provider,
                        settings=settings,
                        mode=mode,
                        attempt=attempt,
                        prepared_directory=staged / f"cell-{ordinal:02d}",
                        command_env=manifest.command_environment,
                    )
                    cell.result_index = len(results)
                    results.append(result)
                    cell.status = "recorded"
                    persist()
                    if result.runtime_config != admission.capture.cells[ordinal - 1].runtime_config:
                        raise ProtocolStop("runtime_configuration")
                    reason = provider.stop_reason or stop_reason(result, plan)
                    if reason:
                        raise ProtocolStop(reason)
                except BaseException as error:
                    primary = error
                    raise
                finally:
                    try:
                        await (provider.aclose() if provider is not None else transport.aclose())
                    except BaseException as error:
                        ledger.cleanup_error = type(error).__name__
                        if primary is None:
                            raise
                        primary.add_note(f"Provider cleanup also failed: {type(error).__name__}")

            # A context failure, including unknown/forced settlement, stops here.
            with owned_backend(
                Path(manifest.image_directory),
                cell_output / "backend",
                Path(manifest.ollama_binary),
            ) as session:
                asyncio.run(execute(session.check_owned))
            check()
        ledger.status = "finished"
        persist()
    except BaseException as error:
        ledger.stop(str(error) if isinstance(error, ProtocolStop) else type(error).__name__)
        try:
            persist()
        except BaseException as recording_error:
            error.add_note(f"Final report also failed: {type(recording_error).__name__}")
        raise
    return ledger
