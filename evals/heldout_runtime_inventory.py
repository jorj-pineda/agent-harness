"""Read-only local interpreter/executable inventory; not an execution freeze."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from itertools import islice
from pathlib import Path

from evals.heldout_inputs import _save
from evals.heldout_model_image import _hash_blob
from evals.heldout_workspaces import _owned_location

MAX_EXECUTABLE_BYTES = 256_000_000


def observe_runtime(executables: dict[str, Path]) -> dict:
    if not 1 <= len(executables) <= 16 or "python" not in executables:
        raise ValueError("Select python and at most sixteen explicit executable paths")
    if executables["python"].resolve(strict=True) != Path(sys.executable).resolve(strict=True):
        raise ValueError("Selected python is not the running interpreter")
    files = {}
    for name, path in executables.items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", name) or not path.is_absolute():
            raise ValueError("Executable labels/paths must be explicit and bounded")
        resolved = path.resolve(strict=True)
        size = resolved.stat().st_size
        if not 1 <= size <= MAX_EXECUTABLE_BYTES:
            raise ValueError("Executable exceeds byte bound")
        pin = _hash_blob(resolved, size)
        if not pin["mode"] & 0o111:
            raise ValueError("Selected file has no executable permission")
        files[name] = {"selected_path": str(path), "resolved_path": str(resolved), **pin}
    packages = {}
    distributions = list(islice(importlib.metadata.distributions(), 2001))
    if len(distributions) > 2000:
        raise ValueError("Installed distribution count exceeds bound")
    for distribution in distributions:
        name, version = distribution.metadata["Name"], distribution.version
        if (
            not name
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+!-]{0,127}", version)
        ):
            raise ValueError("Invalid or unbounded distribution identity")
        name = re.sub(r"[-_.]+", "-", name).lower()
        if name in packages:
            raise ValueError("Ambiguous installed distribution identity")
        packages[name] = version
    return {
        "format": "heldout-runtime-inventory-v1",
        "python_version": list(sys.version_info[:3]),
        "implementation": sys.implementation.name,
        "executables": files,
        "distributions": dict(sorted(packages.items())),
        "dependency_bytes_verified": False,
        "command_environment_verified": False,
        "inference_authorized": False,
    }


def record_runtime(executables: dict[str, Path], report: Path) -> dict:
    if any(not path.is_absolute() for path in executables.values()):
        raise ValueError("Executable paths must be absolute")
    destination = _owned_location(report)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Runtime inventory report already occupied")
    # A selected symlink may itself be the output name, even if its target differs.
    if any(
        destination == path or destination == path.resolve(strict=True)
        for path in executables.values()
    ):
        raise ValueError("Report overlaps executable")
    result = observe_runtime(executables)
    _save(destination, json.dumps(result, indent=2).encode())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", action="append", default=[], metavar="LABEL=ABSOLUTE_PATH")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    selected = {}
    for value in args.executable:
        name, separator, path = value.partition("=")
        if not separator or name in selected:
            raise ValueError("Supply distinct executable labels with explicit paths")
        selected[name] = Path(path)
    record_runtime(selected, args.report)
    print("Runtime inventory retained; dependency bytes, environment and launch remain unverified.")


if __name__ == "__main__":
    main()
