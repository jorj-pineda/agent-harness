"""Independent acceptance. This directory and baseline.json stay outside the workspace."""

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

IGNORED = {".git", "node_modules", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
TARGET = "client/src/lib/format.ts"


def manifest(root: Path) -> dict[str, dict[str, str | int]]:
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        dirs[:] = sorted(name for name in dirs if name not in IGNORED)
        assert all(not (parent / name).is_symlink() for name in dirs)
        for name in sorted(files):
            if name in IGNORED or name == ".env" or name.startswith(".env."):
                continue
            path = parent / name
            metadata = path.lstat()
            assert stat.S_ISREG(metadata.st_mode), f"Unexpected non-regular file: {path.name}"
            result[path.relative_to(root).as_posix()] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "mode": stat.S_IMODE(metadata.st_mode),
            }
    return result


def test_only_requested_file_changes_and_preexisting_work_is_preserved() -> None:
    baseline = json.loads(Path(__file__).with_name("baseline.json").read_text())
    actual = manifest(Path.cwd())
    assert actual.keys() == baseline.keys(), "Unexpected created/deleted files"
    for path, record in baseline.items():
        if path == TARGET:
            assert actual[path]["mode"] == record["mode"]
        else:
            assert actual[path] == record, f"Unrelated change: {path}"


def test_duration_semantics_and_existing_format_contract() -> None:
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            """
import assert from 'node:assert/strict';
import {formatDuration, formatTime, formatTag} from './client/src/lib/format.ts';
for (const seconds of [-0.01, -1, -60, -3600, -1e9, NaN, Infinity, -Infinity]) {
  assert.equal(formatDuration(seconds), '0m');
  assert.equal(formatTime(seconds), '0:00');
}
for (const seconds of [0, 0.1, 1, 5, 59.9, 60, 90, 119, 300, 1500, 3599.9,
                       3600, 3600.9, 3660, 5400, 7200, 86400, 123456.78]) {
  const whole = Math.floor(seconds);
  const minutes = Math.floor(whole / 60);
  const hours = Math.floor(minutes / 60);
  assert.equal(formatDuration(seconds), hours > 0 ? `${hours}h ${minutes % 60}m` : `${minutes}m`);
  assert.equal(formatTime(seconds), `${minutes}:${String(whole % 60).padStart(2, '0')}`);
}
assert.equal(formatTag('Jorge', '0001'), 'Jorge#0001');
assert.equal(formatTag('', ''), '#');
""",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
