"""Workspace path guards, change snapshots, and disposable copies for trusted tasks."""

from __future__ import annotations

from .changes import SnapshotLimitError, WorkspaceChanges, compare, snapshot
from .core import Workspace, WorkspaceError
from .disposable import DisposableWorkspace, disposable_workspace

__all__ = [
    "DisposableWorkspace",
    "SnapshotLimitError",
    "Workspace",
    "WorkspaceChanges",
    "WorkspaceError",
    "compare",
    "disposable_workspace",
    "snapshot",
]
