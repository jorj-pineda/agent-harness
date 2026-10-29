"""Exact structured-fact checks for trusted, public repository questions.

This grades only the requested JSON facts, not prose, citations or comprehension.
Workspace comparison is bounded/scoped and is not process isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.process import ProcessResult
from workspace.changes import snapshot
from workspace.disposable import COPY_IGNORE_PATTERNS

MAX_FACT_BYTES = 16_384


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate fact key")
        result[key] = value
    return result


def canonical_facts(text: str) -> str:
    if len(text.encode("utf-8")) > MAX_FACT_BYTES:
        raise ValueError("Fact object exceeds byte limit")
    value = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(value, dict) or not value:
        raise ValueError("Expected a nonempty fact object")
    # Canonical JSON distinguishes bool/int/string and rejects non-finite values.
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def grade_question(fixture: Path, root: Path, acceptance: Path, answer: str) -> ProcessResult:
    root = root.resolve(strict=True)
    rubric = (acceptance / "facts.json").resolve(strict=True)
    if rubric.is_relative_to(root):
        raise ValueError("Authoritative facts must be outside the editable workspace")
    with rubric.open("rb") as stream:
        expected = canonical_facts(stream.read(MAX_FACT_BYTES + 1).decode("utf-8"))
    modes: list[dict[str, int]] = [{}, {}]
    trees = [
        snapshot(
            path,
            ignore=COPY_IGNORE_PATTERNS,
            max_files=2_000,
            max_bytes=16_000_000,
            strict=True,
            file_modes=modes[index],
        )
        for index, path in enumerate((fixture, root))
    ]
    unchanged = trees[0] == trees[1] and modes[0] == modes[1]
    try:
        matching = canonical_facts(answer) == expected
    except (ValueError, RecursionError):
        matching = False
    passed = matching and unchanged
    return ProcessResult(
        argv=[],  # In-process facts check; no verification command was executed.
        exit_code=0 if passed else 1,
        stdout=(
            f"Structured facts: {'matched' if matching else 'rejected'}; "
            f"scoped workspace: {'unchanged' if unchanged else 'changed'}.\n"
        ),
        stderr="",
        stdout_truncated=False,
        stderr_truncated=False,
    )
