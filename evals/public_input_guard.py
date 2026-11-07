"""Load public first-input pins for the shared evaluation-only provider guard."""

from __future__ import annotations

from pathlib import Path

from evals.heldout_input_guard import InputPins
from evals.public_staging import _pinned, verify_public_capture


def load_public_pins(
    capture: Path,
    *,
    original_directory: Path,
    expected_index_sha256: str,
    ordinal: int,
) -> InputPins:
    """Bind one fixed public cell; no transport, provider or approval is created.

    Archive verification does not observe historical runtime/dependency paths.
    Callers own admission, fresh artifact/resource checks and provider lifecycle.
    Returned bytes can be enforced by the existing PinnedOllamaProvider; later
    requests still require the caller's ongoing runtime and ownership guards.
    """
    if type(ordinal) is not int or not 1 <= ordinal <= 6:
        raise ValueError("Select one of the six public first-input ordinals")
    index = verify_public_capture(
        capture,
        expected_index_sha256=expected_index_sha256,
        original_directory=original_directory,
    )
    cell = index.cells[ordinal - 1]
    contents = [
        _pinned(capture / name, cell.files[name])
        for name in (f"{ordinal:02d}-normalized.json", f"{ordinal:02d}-wire.json")
    ]
    # Refuse a changed selection while extracting the selected request bytes.
    if (
        verify_public_capture(
            capture,
            expected_index_sha256=expected_index_sha256,
            original_directory=original_directory,
        )
        != index
    ):
        raise ValueError("Public capture selection changed while loading first-input pins")
    return InputPins(*contents)
