"""Development-only audit formatting outside a retained evaluation runtime.

This prevents the known Ruff cache/cwd mistake for commands using this entry point.
It does not seal a runtime, sandbox Ruff, or stop unrelated external writers.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def format_audit_script(
    script: Path, *, runtime: Path, working_directory: Path
) -> subprocess.CompletedProcess[str]:
    """Refuse runtime targets/cwd before spawning; keep all existing evidence intact."""
    runtime = runtime.resolve(strict=True)
    working_directory = working_directory.resolve(strict=True)
    script = script.resolve(strict=True)
    if not runtime.is_dir() or not working_directory.is_dir() or not script.is_file():
        raise ValueError("Runtime/cwd must be directories and the audit script a file.")
    if working_directory.is_relative_to(runtime) or script.is_relative_to(runtime):
        raise ValueError("Development cwd and audit script must stay outside the retained runtime.")
    return subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--no-cache", str(script)],
        cwd=working_directory,
        env={**os.environ, "RUFF_NO_CACHE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--working-directory", required=True, type=Path)
    parser.add_argument("script", type=Path)
    args = parser.parse_args()
    try:
        result = format_audit_script(
            args.script, runtime=args.runtime, working_directory=args.working_directory
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
