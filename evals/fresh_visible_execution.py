"""Serial admitted fresh cells with per-cell owned lifecycle; no launch CLI."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import httpx

from evals.fresh_visible_archive import verify_fresh_capture
from evals.fresh_visible_execution_admission import (
    FreshExecutionAdmission,
    check_fresh_admission,
)
from evals.fresh_visible_execution_manifest import FreshExecutionManifest
from evals.fresh_visible_execution_preflight import check_fresh_runtime
from evals.fresh_visible_execution_provider import build_fresh_cell_provider
from evals.fresh_visible_runner import FreshLedger, run_fresh_schedule
from evals.fresh_visible_staging import StagedFresh
from evals.fresh_visible_workspaces import (
    ROOT,
    SELECTION_SHA256,
    FreshCell,
    FreshCellWorkspace,
    selected_cell,
    verify_fresh_cell,
)
from evals.heldout_backend_rehearsal import owned_backend
from evals.heldout_execution_manifest import _selected_bytes
from evals.heldout_guard import _read_regular
from evals.heldout_inputs import check_dependency_selection
from evals.heldout_model_copy import prepare_image
from evals.heldout_model_image import _hash_blob
from evals.heldout_native_evidence import NativeEvidenceTransport
from evals.heldout_operator_rehearsal import _runtime_matches
from evals.heldout_rehearsal import CheckedProvider, ProtocolStop
from evals.heldout_settings import check_settings, load_settings_pin
from evals.public_envelope import _digest, public_settings, validate_public_plan
from evals.public_staging import _inventory, _pinned, _private_directory
from evals.public_workspaces import _destination
from evals.real_run import (
    RealTaskResult,
    _validate_report_path,
    load_tasks,
    run_task,
    validate_task,
    write_report,
)
from evals.residency_observation import ResidencyObservedProvider, record_residency


def _result_stop(result: RealTaskResult, expected_model: str) -> str | None:
    if result.termination.startswith("error:"):
        return "runtime_or_provider_error"
    if result.model != expected_model:
        return "model_identity"
    if result.prompt_tokens is None or result.completion_tokens is None:
        return "missing_usage"
    if (result.acceptance_identity or {}).get("status") != "matching":
        return "criterion_identity"
    if result.runtime_config.get("fresh_visible_preservation", {}).get("status") != "checks_passed":
        return "preservation"
    if result.termination not in {
        "accepted",
        "acceptance_failed",
        "budget_exhausted",
        "iteration_limit",
        "truncated",
        "blocked",
        "incomplete",
    }:
        return "unexpected_terminal_state"
    return None


def _settlement_files(output: Path) -> dict[str, str]:
    contents = {}
    for name in ("image/prepared-image.json", "backend/session.json", "backend/cleanup.json"):
        raw, mode = _read_regular(output / name, 2_000_000)
        if mode != 0o600:
            raise ValueError("Fresh settlement evidence must stay private")
        contents[name] = raw
    cleanup = json.loads(contents["backend/cleanup.json"])
    session = json.loads(contents["backend/session.json"])
    if (
        cleanup.get("passed") is not True
        or cleanup.get("settled") is not True
        or cleanup.get("forced") is not False
        or cleanup.get("returncode") != 0
        or cleanup.get("errors") != {}
        or session.get("format") != "heldout-owned-backend-session-v1"
        or session.get("inference_authorized") is not False
        or session.get("cleanup") != cleanup
    ):
        raise ValueError("Fresh owned settlement did not pass")
    return {name: _digest(raw) for name, raw in contents.items()}


def run_fresh_cohort(
    admission: FreshExecutionAdmission,
    *,
    transport_factory: Callable[[int, Path], httpx.AsyncBaseTransport],
    scripted: bool,
) -> FreshLedger:
    """Trusted admitted caller only; explicit transport, no default approval or retry.

    Record each result before owned cleanup, and advance only after settlement.
    The stopped cell can retain its result while its ledger remains started.
    Scripted mode omits the real GET-only observer; it does not imply real approval.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise ValueError("Enter fresh executor outside an active event loop")
    check_fresh_admission(admission)
    manifest = FreshExecutionManifest.model_validate(admission.manifest.model_dump())
    plan = validate_public_plan()
    cells = [selected_cell(i)[0] for i in range(1, 13)]
    if [row.cell for row in admission.capture.cells] != cells:
        raise ValueError("Admitted fresh cells differ from the fixed schedule")
    staged = Path(manifest.staged_directory)
    capture = Path(manifest.archive_directory) / "captured"
    destination = admission.directory
    if destination != Path(manifest.output_directory):
        raise ValueError("Fresh output differs from admitted selection")
    tasks = load_tasks(ROOT / "tasks.yaml")
    by_id = {task.id: task for task in tasks}
    settings = public_settings()
    expected_settings = load_settings_pin(
        capture / "settings.json", expected_sha256=manifest.settings_sha256
    )
    results: list[RealTaskResult] = []
    current: FreshLedger | None = None
    settlements: dict[Path, dict[str, str]] = {}

    def check() -> None:
        check_fresh_admission(admission)
        selected = FreshExecutionManifest.model_validate_json(
            _selected_bytes(
                destination / "manifest.json", admission.approval.manifest_sha256, 64_000
            )
        )
        if selected != manifest:
            raise ValueError("Fresh effective manifest differs from preserved selection")
        for ordinal in range(1, 13):
            selected_cell(ordinal)
        for prior, pins in settlements.items():
            if _settlement_files(prior) != pins:
                raise ValueError("Fresh earlier settlement evidence changed")
        _runtime_matches({"commit": manifest.runtime_commit, "dirty": False})
        check_settings(settings, expected_settings)
        verify_fresh_capture(
            capture, expected_index_sha256=manifest.capture_index_sha256, original_directory=staged
        )
        check_fresh_runtime(capture / "runtime-inventory.json", manifest.runtime_inventory_sha256)
        check_dependency_selection(
            capture / "dependency-files.json", manifest.dependency_files_sha256
        )
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
            raise ValueError("Fresh selected executable changed")
        _private_directory(staged)
        _inventory(staged, {"staged.json", *(f"cell-{cell.ordinal:02d}" for cell in cells)})
        raw, mode = _read_regular(staged / "staged.json", 4096)
        if mode != 0o600 or StagedFresh.model_validate_json(raw) != StagedFresh(
            selection_sha256=SELECTION_SHA256,
            capture_index_sha256=manifest.capture_index_sha256,
            archive=str(capture),
        ):
            raise ValueError("Fresh staging receipt changed")

    def persist(ledger: FreshLedger) -> None:
        nonlocal current
        current = ledger
        if destination.is_symlink() or _destination(destination) != destination:
            raise ValueError("Fresh output path changed")
        _validate_report_path(destination / "report.json", ROOT / "tasks.yaml", tasks)
        write_report(
            destination / "report.json",
            "public fresh visible-check cohort",
            results,
            cohort=ledger,
            scripted=scripted,
        )

    async def controls() -> None:
        for task in tasks:
            check()
            await validate_task(task, command_env=manifest.command_environment)

    def execute_and_settle(cell: FreshCell) -> None:
        check()
        verify_fresh_cell(staged / f"cell-{cell.ordinal:02d}", ordinal=cell.ordinal)
        if cell.ordinal == 1:
            asyncio.run(controls())
        output = destination / f"cell-{cell.ordinal:02d}"
        output.mkdir(mode=0o700)
        image = output / "image"
        prepare_image(
            Path(manifest.image_directory) / "models",
            image,
            expected_manifest_sha256=plan.model_digest,
        )
        check()

        async def execute(check_owned: Callable[[], None]) -> None:
            def guarded_owned() -> None:
                check()
                check_owned()

            transport = transport_factory(cell.ordinal, output)
            provider: CheckedProvider | ResidencyObservedProvider | None = None
            primary = None
            try:
                transport = NativeEvidenceTransport(transport, output / "native")
                provider = build_fresh_cell_provider(
                    admission,
                    cell.ordinal,
                    settings=settings,
                    transport=transport,
                    check_owned=guarded_owned,
                )
                if not scripted:

                    async def observe(ordinal: int) -> None:
                        await record_residency(
                            output, ordinal, plan.keep_alive_seconds, guarded_owned
                        )

                    provider = ResidencyObservedProvider(provider, observe)
                result = await run_task(
                    by_id[cell.task_id],
                    provider=provider,
                    settings=settings,
                    mode="harness",
                    attempt=cell.repeat,
                    fresh_cell=FreshCellWorkspace(
                        ordinal=cell.ordinal, directory=staged / f"cell-{cell.ordinal:02d}"
                    ),
                    command_env=manifest.command_environment,
                )
                results.append(result)
                assert current is not None
                persist(current)
                expected_config = json.loads(
                    _pinned(
                        capture / f"{cell.ordinal:02d}-runtime-config.json",
                        admission.capture.cells[cell.ordinal - 1].files[
                            f"{cell.ordinal:02d}-runtime-config.json"
                        ],
                    )
                )
                actual = dict(result.runtime_config)
                # Request/grading checkpoint counts are observational, not configured policy.
                expected_config.pop("fresh_visible_preservation", None)
                actual.pop("fresh_visible_preservation", None)
                if actual != expected_config:
                    raise ProtocolStop("runtime_configuration")
                reason = provider.stop_reason or _result_stop(result, plan.model)
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
                    primary.add_note(f"Fresh provider cleanup also failed: {type(error).__name__}")

        with owned_backend(
            image,
            output / "backend",
            Path(manifest.ollama_binary),
            keep_alive_seconds=plan.keep_alive_seconds,
        ) as session:
            asyncio.run(execute(session.check_owned))
        settlements[output] = _settlement_files(output)
        check()

    check()
    for cell in cells:
        verify_fresh_cell(staged / f"cell-{cell.ordinal:02d}", ordinal=cell.ordinal)
    return run_fresh_schedule(
        destination / "cohort-started.json", execute_and_settle=execute_and_settle, persist=persist
    )
