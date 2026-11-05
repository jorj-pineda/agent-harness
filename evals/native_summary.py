"""Read-only, bounded native evidence summary; no provider or execution authority."""

from __future__ import annotations

import argparse
import json
import math
import re
import stat
from itertools import islice
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evals.heldout_guard import _read_regular
from evals.heldout_native_evidence import MAX_BYTES, MAX_REQUESTS

SUFFIXES = (
    "json",
    "request.bin",
    "response.bin",
    "partial-response.bin",
    "failure.json",
    "close-failure.json",
)


class RequestSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    ordinal: int
    status_code: int | None = None
    model_completed: bool | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    elapsed_seconds: float | None = None
    failure_recorded: bool
    partial_response_recorded: bool


class NativeSummary(BaseModel):
    format: Literal["native-evidence-summary-v1"] = "native-evidence-summary-v1"
    requests: list[RequestSummary] = Field(default_factory=list)
    limits: str = (
        "Read-only, non-atomic observations, not acceptance or launch authority. "
        "Null timing/usage is unknown; complete HTTP errors are not model completions. "
        "Payload text omitted. Elapsed scope excludes response closure and outer lifecycle."
    )


def _object(path: Path) -> dict:
    raw, mode = _read_regular(path, MAX_BYTES)
    if mode != 0o600:
        raise ValueError("Native evidence mode changed")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Native metadata must be an object")
    return value


def _count(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= 1_000_000_000 else None


def summarize(directory: Path) -> NativeSummary:
    if directory.is_symlink() or directory.resolve(strict=True) != directory:
        raise ValueError("Select a canonical native evidence directory")
    paths = list(islice(directory.iterdir(), MAX_REQUESTS * len(SUFFIXES) + 1))
    if len(paths) > MAX_REQUESTS * len(SUFFIXES):
        raise ValueError("Native evidence inventory exceeds bound")
    ordinals = set()
    names = set()
    for path in paths:
        match = re.fullmatch(r"request-(\d{2})\.(.+)", path.name)
        info = path.lstat()
        if (
            match is None
            or not 1 <= int(match[1]) <= MAX_REQUESTS
            or match[2] not in SUFFIXES
            or not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > MAX_BYTES
        ):
            raise ValueError("Unexpected or nonprivate native evidence entry")
        ordinals.add(int(match[1]))
        names.add(path.name)
    rows = []
    for ordinal in sorted(ordinals):
        prefix = f"request-{ordinal:02d}"
        metadata = _object(directory / f"{prefix}.json") if f"{prefix}.json" in names else {}
        status = metadata.get("status_code")
        if status is not None and (type(status) is not int or not 100 <= status <= 599):
            raise ValueError("Invalid native HTTP status")
        timing = metadata.get("timing")
        failure = f"{prefix}.failure.json" in names
        if timing is None and failure:
            timing = _object(directory / f"{prefix}.failure.json").get("timing")
        elapsed = None
        if timing is not None:
            if (
                not isinstance(timing, dict)
                or timing.get("scope") != "native_request_to_evidence_record_v1"
            ):
                raise ValueError("Unknown native timing scope")
            elapsed = timing.get("elapsed_seconds")
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Invalid native elapsed time")
            elapsed = float(elapsed)
        completed = None
        prompt = output = None
        if f"{prefix}.response.bin" in names:
            completed = False if status is not None else None
            raw, mode = _read_regular(directory / f"{prefix}.response.bin", MAX_BYTES)
            if mode != 0o600:
                raise ValueError("Native response mode changed")
            try:
                body = json.loads(raw)
                if not isinstance(body, dict):
                    body = {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = {}
            if (
                status is not None
                and 200 <= status < 300
                and body.get("done") is True
                and isinstance(body.get("model"), str)
                and body["model"].strip()
            ):
                completed = True
                prompt, output = (
                    _count(body.get("prompt_eval_count")),
                    _count(body.get("eval_count")),
                )
        rows.append(
            RequestSummary(
                ordinal=ordinal,
                status_code=status,
                model_completed=completed,
                prompt_tokens=prompt,
                completion_tokens=output,
                elapsed_seconds=elapsed,
                failure_recorded=failure or f"{prefix}.close-failure.json" in names,
                partial_response_recorded=f"{prefix}.partial-response.bin" in names,
            )
        )
    return NativeSummary(requests=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(summarize(args.directory).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
