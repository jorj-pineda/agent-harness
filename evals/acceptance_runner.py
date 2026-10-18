"""Owned pytest bootstrap, launched with Python -I -B before adding task imports.

This prevents editable pytest modules/configuration from selecting the grader.
Imported implementation code still runs on the host: this is not process isolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def main() -> int:
    root = Path(sys.argv[1]).resolve(strict=True)
    acceptance = Path(sys.argv[2]).resolve(strict=True)
    if acceptance.is_relative_to(root):
        raise ValueError("Authoritative acceptance must be outside the editable workspace")
    # pytest is already imported from the isolated interpreter's installed packages.
    # Task modules remain importable by the external acceptance suite.
    sys.path.insert(0, str(root))
    return int(
        pytest.main(
            [
                "-q",
                "--noconftest",
                "-c",
                str(Path(__file__).with_name("acceptance_pytest.ini")),
                "--rootdir",
                str(acceptance),
                "-p",
                "no:cacheprovider",
                str(acceptance),
            ]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
