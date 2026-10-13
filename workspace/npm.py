"""Recognize one bounded-purpose npm test script form, without invoking npm."""

from __future__ import annotations

from pathlib import PurePosixPath


def npm_test_prefix(argv: list[str]) -> str | None:
    """Return the lexical repo-relative prefix, or None for an unapproved form.

    The caller must resolve the directory inside its actual workspace before
    execution. This validates invocation shape, not package script contents.
    """
    if not argv or argv[0] != "npm":
        return None
    if any(not arg or "\n" in arg or "\x00" in arg for arg in argv):
        return None
    rest = argv[1:]
    prefix = "."
    if rest[:1] == ["--prefix"]:
        if len(rest) < 2:
            return None
        prefix, rest = rest[1], rest[2:]
        path = PurePosixPath(prefix)
        if (
            path.is_absolute()
            or ".." in path.parts
            or str(path) != prefix
            or "\\" in prefix
            or prefix.startswith("-")
            or ":" in prefix
        ):
            return None
    if rest[:2] != ["run", "test:run"]:
        return None
    tail = rest[2:]
    if tail and tail[0] != "--":
        return None
    # Do not treat help, test enumeration or no-test success as observed checks.
    # Watch forms are excluded so the advertised operation is a one-shot check.
    rejected = {
        "--help",
        "-h",
        "--version",
        "-V",
        "--watch",
        "-w",
        "--list",
        "--passWithNoTests",
        "--collect-only",
    }
    if any(arg.split("=", 1)[0] in rejected for arg in tail[1:]):
        return None
    return prefix
