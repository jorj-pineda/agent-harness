"""Execute a fixed admitted cohort with per-cell owned settlement; no launch CLI."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from pathlib import Path

import httpx

from evals.heldout_backend_rehearsal import owned_backend
from evals.heldout_comparison import PLAN_PATH, validate_plan
from evals.heldout_execution_admission import ExecutionAdmission, check_admission_withdrawal
from evals.heldout_execution_manifest import ExecutionManifest, _selected_bytes
from evals.heldout_execution_provider import build_cell_provider
from evals.heldout_guard import REGISTRATIONS
from evals.heldout_inputs import check_dependency_selection, check_runtime_selection
from evals.heldout_model_copy import prepare_image
from evals.heldout_model_image import _hash_blob
from evals.heldout_native_evidence import NativeEvidenceTransport
from evals.heldout_operator_rehearsal import _runtime_matches, _staged_inventory
from evals.heldout_rehearsal import ProtocolStop, stop_reason
from evals.heldout_settings import protocol_settings
from evals.heldout_staging import verify_residency_capture
from evals.heldout_workspaces import _owned_location, verify_prepared
from evals.real_run import (
    _validate_report_path,
    load_tasks,
    run_task,
    validate_task,
    write_report,
)
from evals.residency_comparison import RESIDENCY_PLAN_PATH, validate_residency_plan
from evals.residency_observation import ResidencyObservedProvider, record_residency
from evals.residency_runner import ResidencyFeasibilityLedger, ResidencyLedger, run_schedule


def run_residency_cohort(
    admission: ExecutionAdmission,
    *,
    transport_factory: Callable[[int, Path], httpx.AsyncBaseTransport],
    scripted: bool,
) -> ResidencyLedger:
    """Run the full fixed cohort; refuse a single-cell feasibility selection."""
    ledger = _run_residency(
        admission, transport_factory=transport_factory, scripted=scripted, feasibility=False
    )
    assert isinstance(ledger, ResidencyLedger)
    return ledger


def run_residency_feasibility(
    admission: ExecutionAdmission,
    *,
    transport_factory: Callable[[int, Path], httpx.AsyncBaseTransport],
    scripted: bool,
) -> ResidencyFeasibilityLedger:
    """Execute only the explicitly approved fresh ISO-week 300-second cell once."""
    ledger = _run_residency(
        admission, transport_factory=transport_factory, scripted=scripted, feasibility=True
    )
    assert isinstance(ledger, ResidencyFeasibilityLedger)
    return ledger


def _run_residency(
    admission: ExecutionAdmission,
    *,
    transport_factory: Callable[[int, Path], httpx.AsyncBaseTransport],
    scripted: bool,
    feasibility: bool,
) -> ResidencyLedger | ResidencyFeasibilityLedger:
    """Trusted admitted caller only; explicit transport, no default launch or approval.

    Results are recorded before cleanup; a ledger cell is recorded only after
    clean settlement. Failed cells may therefore have a result while still started.
    Result rows follow schedule ordinal order, including the stopped final cell.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise ValueError("Enter residency executor outside an active event loop")
    if not scripted and getattr(admission, "source_manifest", None) is None:
        raise ValueError("Real residency execution requires selected source provenance")
    manifest = ExecutionManifest.model_validate(admission.manifest.model_dump())
    if (manifest.residency_feasibility_ordinal is not None) != feasibility:
        raise ValueError("Use the matching explicit feasibility or full-cohort executor")
    selected = validate_residency_plan()
    if (
        manifest.residency_plan_sha256
        != hashlib.sha256(RESIDENCY_PLAN_PATH.read_bytes()).hexdigest()
        or manifest.keep_alive_seconds != 0
    ):
        raise ValueError("Explicit matching residency schedule and no global arm override required")
    if [(c.task_id, c.mode, c.attempt, c.keep_alive_seconds) for c in admission.capture.cells] != [
        (c.task_id, c.mode, c.repeat, c.keep_alive_seconds) for c in selected.cells
    ]:
        raise ValueError("Admitted cells differ from residency schedule")
    staged, capture = Path(manifest.staged_directory), Path(manifest.capture_directory)
    destination = admission.directory
    plan = validate_plan()
    tasks = load_tasks(REGISTRATIONS[0].root / "tasks.yaml")
    by_id = {task.id: task for task in tasks}
    settings = protocol_settings(plan, prompt_profile=manifest.coding_prompt_profile)
    first_ordinal = 2 if feasibility else 1
    results = []
    current = None

    def check() -> None:
        check_admission_withdrawal(admission)
        if (
            hashlib.sha256(RESIDENCY_PLAN_PATH.read_bytes()).hexdigest()
            != manifest.residency_plan_sha256
        ):
            raise ValueError("Residency plan changed")
        validate_residency_plan()
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
        verify_residency_capture(
            capture, expected_index_sha256=manifest.capture_index_sha256, original_directory=staged
        )
        check_runtime_selection(
            capture / "runtime-inventory.json", manifest.runtime_inventory_sha256
        )
        check_dependency_selection(
            capture / "dependency-files.json", manifest.dependency_files_sha256
        )
        _staged_inventory(
            staged, capture, manifest.capture_index_sha256, admission.capture.cells, residency=True
        )

    def persist(ledger):
        nonlocal current
        current = ledger
        if (
            destination.is_symlink()
            or destination.resolve() != destination
            or _owned_location(destination) != destination
        ):
            raise ValueError("Residency output path changed")
        _validate_report_path(
            destination / "report.json", REGISTRATIONS[0].root / "tasks.yaml", tasks
        )
        write_report(
            destination / "report.json",
            "residency development feasibility" if feasibility else "residency development cohort",
            results,
            cohort=ledger,
            scripted=scripted,
        )

    async def controls():
        for task in tasks:
            check()
            await validate_task(task, command_env=manifest.command_environment)

    def execute_and_settle(cell):
        check()
        if cell.ordinal == first_ordinal:
            asyncio.run(controls())
        task = by_id[cell.task_id]
        output = destination / f"cell-{cell.ordinal:02d}"
        output.mkdir(mode=0o700)
        # Each cell owns a fresh independently verified image; only the immutable
        # selected source bundle is shared. Preparation failure preserves evidence
        # and stops the schedule before backend or transport construction.
        image = output / "image"
        prepare_image(
            Path(manifest.image_directory) / "models",
            image,
            expected_manifest_sha256=plan.model_digest,
        )
        check()

        async def execute(check_owned):
            def guarded_owned():
                check()
                check_owned()

            transport = transport_factory(cell.ordinal, output)
            provider = None
            primary = None
            try:
                transport = NativeEvidenceTransport(transport, output / "native")
                provider = build_cell_provider(
                    admission,
                    cell.ordinal,
                    settings=settings,
                    transport=transport,
                    check_owned=guarded_owned,
                )
                if not scripted:

                    async def observe(ordinal):
                        await record_residency(
                            output, ordinal, cell.keep_alive_seconds, guarded_owned
                        )

                    provider = ResidencyObservedProvider(provider, observe)
                result = await run_task(
                    task,
                    provider=provider,
                    settings=settings,
                    mode=cell.mode,
                    attempt=cell.repeat,
                    prepared_directory=staged / f"cell-{cell.ordinal:02d}",
                    command_env=manifest.command_environment,
                )
                results.append(result)
                persist(current)
                if (
                    result.runtime_config
                    != admission.capture.cells[cell.ordinal - 1].runtime_config
                ):
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
                    if primary is None:
                        raise
                    primary.add_note(f"Provider cleanup also failed: {type(error).__name__}")

        with owned_backend(
            image,
            output / "backend",
            Path(manifest.ollama_binary),
            keep_alive_seconds=cell.keep_alive_seconds,
        ) as session:
            asyncio.run(execute(session.check_owned))
        check()

    check()
    for pinned in admission.capture.cells:
        verify_prepared(staged / f"cell-{pinned.ordinal:02d}", task_id=pinned.task_id)
    return run_schedule(
        destination / "cohort-started.json",
        execute_and_settle=execute_and_settle,
        persist=persist,
        feasibility=feasibility,
    )
