"""Read-only pins for explicitly selected trusted dependency code/library files."""

from __future__ import annotations

import argparse
import hashlib
import stat
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evals.heldout_guard import _read_regular
from evals.heldout_inputs import _save
from evals.heldout_model_image import _hash_blob
from evals.heldout_workspaces import _owned_location

MAX_FILE_BYTES = 256_000_000
MAX_TOTAL_BYTES = 2_000_000_000
SUFFIXES = {".py", ".pyc", ".so", ".dylib", ".dll", ".pyd", ".pth"}


class DependencyFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0, le=MAX_FILE_BYTES)
    mode: int = Field(ge=0, le=0o7777)


class DependencyFiles(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    format: Literal["heldout-selected-dependency-files-v1"] = "heldout-selected-dependency-files-v1"
    files: dict[str, DependencyFile] = Field(min_length=1, max_length=4096)
    import_closure_verified: Literal[False] = False
    inference_authorized: Literal[False] = False


def _selected_path(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("Dependency selection requires an absolute non-link file")
    resolved = path.resolve(strict=True)
    for candidate in (path, resolved):
        if candidate.suffix not in SUFFIXES or any(
            part == ".git" or part == ".env" or part.startswith(".env.") for part in candidate.parts
        ):
            raise ValueError("Select trusted dependency code/library files, not configuration")
    return resolved


def observe_files(paths: list[Path]) -> DependencyFiles:
    if not 1 <= len(paths) <= 4096:
        raise ValueError("Dependency file count exceeds bound")
    selected = [_selected_path(path) for path in paths]
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate dependency file selection")
    sizes = {}
    for path in selected:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= MAX_FILE_BYTES:
            raise ValueError("Invalid dependency file type/size")
        sizes[path] = info.st_size
    if sum(sizes.values()) > MAX_TOTAL_BYTES:
        raise ValueError("Dependency files exceed aggregate byte bound")
    return DependencyFiles(
        files={str(path): DependencyFile(**_hash_blob(path, size)) for path, size in sizes.items()}
    )


def record_files(paths: list[Path], report: Path) -> DependencyFiles:
    if not 1 <= len(paths) <= 4096:
        raise ValueError("Dependency file count exceeds bound")
    destination = _owned_location(report)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Dependency file report already occupied")
    if any(destination == path or destination == path.resolve(strict=True) for path in paths):
        raise ValueError("Dependency report overlaps selected file")
    result = observe_files(paths)
    _save(destination, result.model_dump_json(indent=2).encode())
    return result


def verify_files(report: Path, *, expected_sha256: str) -> DependencyFiles:
    raw, mode = _read_regular(report, 2_000_000)
    if mode != 0o600 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Dependency file report bytes/mode changed")
    expected = DependencyFiles.model_validate_json(raw)
    paths = [Path(name) for name in expected.files]
    # Recorded paths are canonical; do not silently resolve a changed parent link.
    if any(_selected_path(path) != path for path in paths):
        raise ValueError("Recorded dependency path target changed")
    if observe_files(paths) != expected:
        raise ValueError("Selected dependency files changed; no rebinding permitted")
    again, again_mode = _read_regular(report, 2_000_000)
    if again != raw or again_mode != mode:
        raise ValueError("Dependency file report changed during verification")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", action="append", type=Path, default=[])
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--verify-sha256")
    args = parser.parse_args()
    if args.verify_sha256 is not None:
        if args.file:
            raise ValueError("Verification cannot replace the selected file list")
        verify_files(args.report, expected_sha256=args.verify_sha256)
    else:
        record_files(args.file, args.report)
    print("Selected dependency file bytes checked; import closure and launch remain unverified.")


if __name__ == "__main__":
    main()
