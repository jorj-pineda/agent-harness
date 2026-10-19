"""Visible library smoke; authoritative acceptance remains external."""

import subprocess
from pathlib import Path


def test_requested_behavior():
    script = Path(__file__).with_name("visible_check.mjs").read_text()
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
