"""Guard composition for an admitted cell; no default network transport or launch CLI."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx

from evals.heldout_comparison import validate_plan
from evals.heldout_execution_admission import ExecutionAdmission
from evals.heldout_execution_manifest import _selected_bytes
from evals.heldout_input_guard import PinnedOllamaProvider, load_pins
from evals.heldout_inputs import check_dependency_selection, check_runtime_selection
from evals.heldout_operator_rehearsal import _runtime_matches
from evals.heldout_rehearsal import CheckedProvider
from evals.heldout_settings import SettingsPinnedProvider, check_settings, load_settings_pin
from evals.owned_inference_transport import OwnedInferenceTransport
from harness.config import Settings


def build_cell_provider(
    admission: ExecutionAdmission,
    ordinal: int,
    *,
    settings: Settings,
    transport: httpx.AsyncBaseTransport,
    check_owned: Callable[[], None],
) -> CheckedProvider:
    """Caller owns valid admission, resource lifecycle and closing returned provider.

    Construction failure leaves transport ownership with the caller. On success,
    closing the returned provider closes the supplied transport. No retry, task
    admission or settlement is performed by this factory.
    """
    manifest = admission.manifest.model_copy(deep=True)
    plan = validate_plan()
    capture = Path(manifest.capture_directory)
    if admission.capture.settings_sha256 is None:
        raise ValueError("Admitted capture lacks Settings pin")
    expected = load_settings_pin(
        capture / "settings.json", expected_sha256=admission.capture.settings_sha256
    )

    def guard() -> None:
        _selected_bytes(
            admission.directory / "manifest.json", admission.approval.manifest_sha256, 64_000
        )
        _runtime_matches({"commit": manifest.runtime_commit, "dirty": False})
        check_settings(settings, expected)
        load_settings_pin(
            capture / "settings.json", expected_sha256=admission.capture.settings_sha256
        )
        check_runtime_selection(
            capture / "runtime-inventory.json", manifest.runtime_inventory_sha256
        )
        check_dependency_selection(
            capture / "dependency-files.json", manifest.dependency_files_sha256
        )
        check_owned()

    guard()
    pins = load_pins(
        capture,
        original_directory=Path(manifest.staged_directory),
        expected_index_sha256=manifest.capture_index_sha256,
        ordinal=ordinal,
    )
    return CheckedProvider(
        SettingsPinnedProvider(
            PinnedOllamaProvider(
                pins=pins,
                host="http://127.0.0.1:11447",
                model=plan.model,
                num_ctx=plan.runtime.ollama_num_ctx,
                think=plan.runtime.ollama_think,
                timeout_seconds=plan.runtime.request_timeout_seconds,
                transport=OwnedInferenceTransport(transport, guard),
            ),
            settings,
            expected,
        ),
        plan,
    )
