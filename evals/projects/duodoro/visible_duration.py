"""Additional visible task checks; copy as test_visible.py before taking the baseline."""

import subprocess


def test_invalid_durations_render_as_zero() -> None:
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            "import assert from 'node:assert/strict'; "
            "import {formatDuration, formatTime} from './client/src/lib/format.ts'; "
            "for (const seconds of [-1, NaN, Infinity]) { "
            "assert.equal(formatDuration(seconds), '0m'); "
            "assert.equal(formatTime(seconds), '0:00'); }",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
