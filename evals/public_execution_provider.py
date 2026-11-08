"""Public admitted provider guard composition; no default transport or execution loop."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx

from evals.heldout_execution_manifest import _selected_bytes
from evals.heldout_input_guard import PinnedOllamaProvider
from evals.heldout_inputs import check_dependency_selection, check_runtime_selection
from evals.heldout_operator_rehearsal import _runtime_matches
from evals.heldout_rehearsal import CheckedProvider
from evals.heldout_settings import SettingsPinnedProvider, check_settings, load_settings_pin
from evals.owned_inference_transport import OwnedInferenceTransport
from evals.public_envelope import validate_public_plan
from evals.public_execution_admission import PublicExecutionAdmission, check_public_admission
from evals.public_execution_manifest import PublicExecutionManifest
from evals.public_input_guard import load_public_pins
from harness.config import Settings


def build_public_cell_provider(
    admission: PublicExecutionAdmission,
    ordinal: int,
    *,
    settings: Settings,
    transport: httpx.AsyncBaseTransport,
    check_owned: Callable[[], None],
) -> CheckedProvider:
    """Caller owns actual trusted admission, lifecycle, task guards and settlement.

    No provider request is made during construction. On failure the caller owns
    transport cleanup; on success closing the returned provider closes transport.
    This factory neither authenticates approval nor creates a backend or retry.
    """
    manifest = PublicExecutionManifest.model_validate(admission.manifest.model_dump())
    capture = Path(manifest.archive_directory) / "captured"
    plan = validate_public_plan()
    expected = load_settings_pin(
        capture / "settings.json", expected_sha256=manifest.settings_sha256
    )

    def guard() -> None:
        check_public_admission(admission)
        preserved = PublicExecutionManifest.model_validate_json(
            _selected_bytes(
                admission.directory / "manifest.json", admission.approval.manifest_sha256, 64_000
            )
        )
        if preserved != manifest:
            raise ValueError("Public provider selection differs from admitted manifest bytes")
        _runtime_matches({"commit": manifest.runtime_commit, "dirty": False})
        check_settings(settings, expected)
        load_settings_pin(capture / "settings.json", expected_sha256=manifest.settings_sha256)
        check_runtime_selection(
            capture / "runtime-inventory.json", manifest.runtime_inventory_sha256
        )
        check_dependency_selection(
            capture / "dependency-files.json", manifest.dependency_files_sha256
        )
        check_owned()

    guard()
    num_ctx, think = settings.ollama_num_ctx, settings.ollama_think
    if num_ctx is None or think is None:
        raise ValueError("Public provider requires explicit context and thinking settings")
    pins = load_public_pins(
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
                num_ctx=num_ctx,
                think=think,
                timeout_seconds=settings.request_timeout_seconds,
                transport=OwnedInferenceTransport(transport, guard),
            ),
            settings,
            expected,
        ),
        plan,
    )
