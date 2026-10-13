"""Visible calendar feature checks; scoped Node resolver, no project build."""

import subprocess


def test_sunday_first_preserves_monday_default():
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            """import assert from 'node:assert/strict';
import {registerHooks} from 'node:module';
registerHooks({resolve(specifier, context, nextResolve) {
  if (specifier === './focusTags' && context.parentURL?.endsWith('/focusCalendar.ts')) {
    return nextResolve('./focusTags.ts', context);
  }
  return nextResolve(specifier, context);
}});
const c = await import('./client/src/lib/focusCalendar.ts');
assert.equal(c.monthDays('2026-02', true)[0], '2026-02-01');
assert.deepEqual(c.monthDays('2024-02').slice(0,4), [null,null,null,'2024-02-01']);
assert.equal(c.monthDays('2024-02',true).filter(Boolean).length,29);
""",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
