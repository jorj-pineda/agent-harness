"""Visible pure-parser task checks; full server integration is outside this adapter."""

import subprocess


def test_fractional_timer_settings_use_whole_seconds():
    result = subprocess.run(
        [
            "node",
            "-e",
            """
const assert = require('node:assert/strict');
const {parseStartSession} = require('./server/payloadParsers.js');
const result = parseStartSession({sessionId: 'room', focusDuration: 1500.9,
                                 breakDuration: '300.99', mode: 'flow'});
assert.deepEqual(result, {ok: true, value: {sessionId: 'room', focusDuration: 1500,
                                          breakDuration: 300, mode: 'flow'}});
assert.equal(parseStartSession({sessionId: 'room',
  focusDuration: Infinity}).value.focusDuration, 7200);
""",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
