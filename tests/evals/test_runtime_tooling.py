"""Real Ruff regression for accidental cache creation in a retained runtime."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evals.runtime_tooling import format_audit_script


def snapshot(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_real_formatter_avoids_runtime_cache_and_preserves_existing_evidence(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    working = tmp_path / "tooling"
    working.mkdir()
    script = working / "audit.py"
    script.write_text("x=1\n")
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop("RUFF_NO_CACHE", None)
    environment.pop("RUFF_CACHE_DIR", None)
    # Reproduce the original mistake on a disposable stand-in, never retained evidence.
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", str(script)],
        cwd=runtime,
        env=environment,
        check=True,
        capture_output=True,
    )
    assert (runtime / ".ruff_cache").is_dir()
    before = snapshot(runtime)
    script.write_text("x=2\n")
    result = format_audit_script(script, runtime=runtime, working_directory=working)
    assert result.returncode == 0 and script.read_text() == "x = 2\n"
    assert snapshot(runtime) == before  # Existing failed-freeze evidence is never cleaned.
    assert not (working / ".ruff_cache").exists()


@pytest.mark.parametrize("bad", ["cwd", "nested_cwd", "script", "symlink"])
def test_refuses_runtime_paths_before_spawning(tmp_path, monkeypatch, bad):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    nested = runtime / "nested"
    nested.mkdir()
    working = tmp_path / "tooling"
    working.mkdir()
    script = working / "audit.py"
    script.write_text("x=1\n")
    if bad in ("script", "symlink"):
        inside = runtime / "inside.py"
        inside.write_text("x=1\n")
        if bad == "symlink":
            script.unlink()
            script.symlink_to(inside)
        else:
            script = inside
    cwd = runtime if bad == "cwd" else nested if bad == "nested_cwd" else working

    def forbidden(*args, **kwargs):
        pytest.fail("Unsafe formatter spawned")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="outside"):
        format_audit_script(script, runtime=runtime, working_directory=cwd)


def test_cli_formats_from_explicit_external_directory(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "pin").write_text("unchanged")
    working = tmp_path / "tooling"
    working.mkdir()
    script = working / "audit.py"
    script.write_text("x=3\n")
    before = snapshot(runtime)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evals.runtime_tooling",
            "--runtime",
            str(runtime),
            "--working-directory",
            str(working),
            str(script),
        ],
        cwd=working,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert script.read_text() == "x = 3\n"
    assert snapshot(runtime) == before
    assert not (working / ".ruff_cache").exists()
