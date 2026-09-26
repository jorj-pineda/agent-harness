"""Workspace path guards and disposable copies for trusted tasks."""

from __future__ import annotations

from .core import Workspace, WorkspaceError
from .disposable import DisposableWorkspace, WorkspaceChanges, disposable_workspace

__all__ = [
    "DisposableWorkspace",
    "Workspace",
    "WorkspaceChanges",
    "WorkspaceError",
    "disposable_workspace",
]
