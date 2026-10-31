"""Capture-bound command environment values; not executable-byte verification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evals.heldout_guard import _read_regular
from evals.real_run import _task_command_environment


class EnvironmentPin(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    format: Literal["heldout-command-environment-v1"] = "heldout-command-environment-v1"
    values: dict[str, str] = Field(max_length=10)


def environment_bytes(value: Mapping[str, str]) -> bytes:
    selected = _task_command_environment(value)
    assert selected is not None
    raw = json.dumps(EnvironmentPin(values=selected).model_dump(), sort_keys=True).encode()
    if len(raw) > 64_000:
        raise ValueError("Command environment pin exceeds byte bound")
    return raw


def load_environment_pin(path: Path, expected_sha256: str) -> bytes:
    raw, mode = _read_regular(path, 64_000)
    if mode != 0o600 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Command environment pin bytes/mode changed")
    pin = EnvironmentPin.model_validate_json(raw)
    if environment_bytes(pin.values) != raw:
        raise ValueError("Command environment pin is not canonical/effective")
    return raw
