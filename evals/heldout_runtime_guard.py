"""Recheck selected runtime inventory without rebinding or granting launch authority."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evals.heldout_guard import _read_regular
from evals.heldout_runtime_inventory import MAX_EXECUTABLE_BYTES, observe_runtime


class InventoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


class ExecutablePin(InventoryModel):
    selected_path: str = Field(min_length=1, max_length=4096)
    resolved_path: str = Field(min_length=1, max_length=4096)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=1, le=MAX_EXECUTABLE_BYTES)
    mode: int = Field(ge=0, le=0o7777)


class RuntimeInventory(InventoryModel):
    format: Literal["heldout-runtime-inventory-v1"]
    python_version: list[int] = Field(min_length=3, max_length=3)
    implementation: str = Field(min_length=1, max_length=128)
    executables: dict[str, ExecutablePin] = Field(min_length=1, max_length=16)
    distributions: dict[str, str] = Field(max_length=2000)
    dependency_bytes_verified: Literal[False]
    command_environment_verified: Literal[False]
    inference_authorized: Literal[False]


def verify_runtime(report: Path, *, expected_sha256: str) -> RuntimeInventory:
    raw, mode = _read_regular(report, 2_000_000)
    if mode != 0o600 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Runtime inventory bytes/mode differ from selected hash")
    expected = RuntimeInventory.model_validate_json(raw)
    selected = {}
    for name, pin in expected.executables.items():
        path = Path(pin.selected_path)
        resolved = Path(pin.resolved_path)
        if not path.is_absolute() or not resolved.is_absolute() or not pin.mode & 0o111:
            raise ValueError("Runtime inventory executable path/mode is invalid")
        selected[name] = path
    observed = observe_runtime(selected)
    if observed != expected.model_dump():
        raise ValueError("Runtime differs from selected inventory; no rebinding permitted")
    again, again_mode = _read_regular(report, 2_000_000)
    if again != raw or again_mode != mode:
        raise ValueError("Runtime inventory changed during verification")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    verify_runtime(args.inventory, expected_sha256=args.sha256)
    print("Runtime inventory matches; dependency bytes, environment and launch remain unverified.")


if __name__ == "__main__":
    main()
