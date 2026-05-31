"""Sandboxed workspace root — all code-tool paths resolve under here."""

from __future__ import annotations

from .core import Workspace, WorkspaceError

__all__ = ["Workspace", "WorkspaceError"]
