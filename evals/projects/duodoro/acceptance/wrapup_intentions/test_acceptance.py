"""External library acceptance; trusted host execution, not process isolation."""

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

IGNORED = {".git", "node_modules", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
TARGETS = ["client/src/lib/sessionWrapUp.ts"]


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


def test_scope_and_preexisting_work():
    baseline = json.loads(Path(__file__).with_name("baseline.json").read_text())
    actual = manifest(Path.cwd())
    assert actual.keys() == baseline.keys(), "Unexpected created/deleted files"
    for path, record in baseline.items():
        if path in TARGETS:
            assert actual[path]["mode"] == record["mode"]
        else:
            assert actual[path] == record, f"Unrelated change: {path}"


def test_behavior_and_preserved_exports():
    env = dict(os.environ)
    env["DUODORO_GOLDEN"] = str(Path(__file__).with_name("golden.json"))
    result = subprocess.run(
        ["node", "--input-type=module", "-e", Path(__file__).with_name("checks.mjs").read_text()],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr[-4000:]
