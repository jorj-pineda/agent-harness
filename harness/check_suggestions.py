"""Bounded static project-check hints; never configuration or execution permission."""

from __future__ import annotations

import configparser
import json
import os
import stat
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

MANIFEST_LIMIT_BYTES = 65_536
ManifestName = Literal["pytest.ini", "pyproject.toml", "package.json"]
MANIFESTS: tuple[ManifestName, ...] = ("pytest.ini", "pyproject.toml", "package.json")


class CheckSuggestion(BaseModel):
    argv: list[str]
    evidence: list[ManifestName]


class ManifestIssue(BaseModel):
    path: ManifestName
    code: Literal["unreadable_or_unsafe", "too_large", "changed", "invalid_manifest"]


class CheckSuggestions(BaseModel):
    status: Literal["inspected", "partial", "unavailable", "busy", "no_workspace"]
    session_id: str | None = None
    suggestions: list[CheckSuggestion] = Field(default_factory=list)
    issues: list[ManifestIssue] = Field(default_factory=list)
    reason: str | None = None
    read_only: Literal[True] = True
    commands_executed: Literal[False] = False
    configuration_changed: Literal[False] = False
    scope_note: str = (
        "Static hints from root pytest.ini, pyproject.toml pytest configuration and package.json "
        "test:run only; no recursion or script evaluation. Commands may be missing, unsafe or "
        "irrelevant. Inspect project setup and explicitly configure PROJECT_CHECK_ARGV before "
        "relying on a suggestion. No command ran and no configuration changed. Reads are bounded, "
        "not an atomic filesystem snapshot, process isolation or execution permission."
    )


class _ManifestReadError(Exception):
    def __init__(self, code: Literal["unreadable_or_unsafe", "too_large", "changed"]) -> None:
        self.code = code


def _signature(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _read(root_fd: int, name: ManifestName) -> str | None:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _ManifestReadError("unreadable_or_unsafe") from exc
    with os.fdopen(fd, "rb", buffering=0) as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise _ManifestReadError("unreadable_or_unsafe")
        data = source.read(MANIFEST_LIMIT_BYTES + 1)
        if len(data) > MANIFEST_LIMIT_BYTES:
            raise _ManifestReadError("too_large")
        after = os.fstat(source.fileno())
        current = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if (
            len(data) != before.st_size
            or _signature(before) != _signature(after)
            or _signature(after) != _signature(current)
        ):
            raise _ManifestReadError("changed")
    return data.decode("utf-8")


def _pytest_marker(name: ManifestName, text: str) -> bool:
    if name == "pytest.ini":
        config = configparser.RawConfigParser()
        config.read_string(text)
        return config.has_section("pytest")
    document = tomllib.loads(text)
    tool = document.get("tool")
    pytest = tool.get("pytest") if isinstance(tool, dict) else None
    return isinstance(pytest, dict) and isinstance(pytest.get("ini_options"), dict)


def suggest_project_checks(root: str | None) -> CheckSuggestions:
    """Inspect at most three fixed root files, each at most 64 KiB plus one sentinel byte."""
    if root is None:
        return CheckSuggestions(status="no_workspace", reason="Session has no workspace.")
    if not hasattr(os, "O_NOFOLLOW") or os.open not in os.supports_dir_fd:
        return CheckSuggestions(
            status="unavailable", reason="Descriptor-relative reads unavailable."
        )
    try:
        root_fd = os.open(Path(root), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return CheckSuggestions(status="unavailable", reason="Workspace directory unavailable.")
    suggestions: list[CheckSuggestion] = []
    issues: list[ManifestIssue] = []
    pytest_evidence: list[ManifestName] = []
    try:
        for name in MANIFESTS:
            try:
                text = _read(root_fd, name)
                if text is None:
                    continue
                if name != "package.json":
                    if _pytest_marker(name, text):
                        pytest_evidence.append(name)
                else:
                    document = json.loads(text)
                    scripts = document.get("scripts") if isinstance(document, dict) else None
                    command = scripts.get("test:run") if isinstance(scripts, dict) else None
                    if isinstance(command, str) and command.strip():
                        suggestions.append(
                            CheckSuggestion(argv=["npm", "run", "test:run"], evidence=[name])
                        )
            except _ManifestReadError as exc:
                issues.append(ManifestIssue(path=name, code=exc.code))
            except (OSError, UnicodeError):
                issues.append(ManifestIssue(path=name, code="unreadable_or_unsafe"))
            except (ValueError, configparser.Error, RecursionError):
                issues.append(ManifestIssue(path=name, code="invalid_manifest"))
        opened = os.fstat(root_fd)
        current = os.stat(root, follow_symlinks=False)
        if not stat.S_ISDIR(current.st_mode) or (opened.st_dev, opened.st_ino) != (
            current.st_dev,
            current.st_ino,
        ):
            return CheckSuggestions(status="unavailable", reason="Workspace identity changed.")
    except OSError:
        return CheckSuggestions(status="unavailable", reason="Workspace identity unavailable.")
    finally:
        os.close(root_fd)
    if pytest_evidence:
        suggestions.insert(
            0, CheckSuggestion(argv=["python", "-m", "pytest"], evidence=pytest_evidence)
        )
    return CheckSuggestions(
        status="partial" if issues else "inspected", suggestions=suggestions, issues=issues
    )
