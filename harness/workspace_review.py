"""Durable, scoped workspace comparison evidence; never resume permission."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from workspace import SnapshotLimitError, compare, snapshot
from workspace.core import DEFAULT_IGNORE_GLOBS

ENV_IGNORE = (".env", ".env.*")
REVIEW_IGNORE = (*DEFAULT_IGNORE_GLOBS, *ENV_IGNORE)


class WorkspaceReviewBaseline(BaseModel):
    schema_version: Literal[1] = 1
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: Literal["recorded", "unavailable", "not_tracked"]
    root: str | None
    root_identity: tuple[int, int] | None = None
    fingerprints: dict[str, str] = Field(default_factory=dict)
    excluded_patterns: tuple[str, ...] = REVIEW_IGNORE
    reason: str | None = None


class WorkspaceReview(BaseModel):
    status: Literal["unchanged", "changed", "unavailable", "busy", "not_recorded", "not_tracked"]
    turn_id: str | None = None
    baseline_captured_at: datetime | None = None
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    added: list[str] = Field(default_factory=list)
    modified: list[str] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)
    added_count: int = 0
    modified_count: int = 0
    deleted_count: int = 0
    paths_truncated: bool = False
    excluded_patterns: tuple[str, ...] = REVIEW_IGNORE
    reason: str | None = None
    read_only: Literal[True] = True
    scope_note: str = (
        "Compares observed non-excluded file contents and symlink targets, not modes or ignored "
        "directory contents. This non-atomic review is not permission to resume or revert."
    )


def review_fingerprints(fingerprints: dict[str, str]) -> dict[str, str]:
    """Never persist environment hashes or raw symlink target strings."""
    return {
        path: hashlib.sha256(value.encode()).hexdigest()
        for path, value in fingerprints.items()
        if not any(fnmatch(part, pattern) for part in Path(path).parts for pattern in ENV_IGNORE)
    }


def compare_workspace(
    baseline: WorkspaceReviewBaseline,
    *,
    max_files: int,
    max_bytes: int,
    max_paths: int = 100,
) -> WorkspaceReview:
    result = WorkspaceReview(
        status="unavailable",
        baseline_captured_at=baseline.captured_at,
        excluded_patterns=baseline.excluded_patterns,
    )
    if baseline.status != "recorded":
        result.status = baseline.status
        result.reason = baseline.reason
        return result
    if baseline.root is None or baseline.root_identity is None:
        result.reason = "Saved workspace identity is unavailable."
        return result
    root = Path(baseline.root)
    try:
        # Reject relocated/replaced roots rather than inspecting a new workspace.
        stat = root.stat()
        if root.resolve() != root or (stat.st_dev, stat.st_ino) != baseline.root_identity:
            result.reason = "Workspace root was replaced or redirected."
            return result
        fingerprints = snapshot(
            root,
            ignore=baseline.excluded_patterns,
            max_files=max_files,
            max_bytes=max_bytes,
            strict=True,
        )
        final_stat = root.stat()
        if (
            root.resolve() != root
            or (final_stat.st_dev, final_stat.st_ino) != baseline.root_identity
        ):
            result.reason = "Workspace root changed during comparison."
            return result
    except (OSError, SnapshotLimitError) as exc:
        result.reason = f"Workspace comparison unavailable ({type(exc).__name__})."
        return result
    changes = compare(baseline.fingerprints, review_fingerprints(fingerprints))
    result.status = (
        "changed" if any((changes.added, changes.modified, changes.deleted)) else "unchanged"
    )
    result.added_count = len(changes.added)
    result.modified_count = len(changes.modified)
    result.deleted_count = len(changes.deleted)
    remaining = max_paths
    for kind in ("added", "modified", "deleted"):
        paths = getattr(changes, kind)
        setattr(result, kind, list(paths[:remaining]))
        remaining = max(0, remaining - len(paths))
    result.paths_truncated = (
        sum((result.added_count, result.modified_count, result.deleted_count)) > max_paths
    )
    return result
