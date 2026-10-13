"""Independent acceptance. This directory and baseline.json stay outside the workspace."""

import calendar
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

IGNORED = {".git", "node_modules", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
TARGET = "client/src/lib/focusCalendar.ts"


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


def test_complete_calendar_grids_and_other_exports():
    cases = []
    for year in [1900, 1999, 2000, 2023, 2024, 2026, 2100]:
        for month in range(1, 13):
            name = f"{year:04d}-{month:02d}"
            for sunday in [False, True]:
                weeks = calendar.Calendar(6 if sunday else 0).monthdayscalendar(year, month)
                expected = [f"{name}-{day:02d}" if day else None for week in weeks for day in week]
                cases.append({"month": name, "sunday": sunday, "expected": expected})
    prefix = """import assert from 'node:assert/strict';
import {registerHooks} from 'node:module';
registerHooks({resolve(specifier, context, nextResolve) {
  if (specifier === './focusTags' && context.parentURL?.endsWith('/focusCalendar.ts')) {
    return nextResolve('./focusTags.ts', context);
  }
  return nextResolve(specifier, context);
}});
const c = await import('./client/src/lib/focusCalendar.ts');
"""
    checks = """
for (const {month,sunday,expected} of cases) {
  assert.deepEqual(c.monthDays(month,sunday),expected);
  if (!sunday) assert.deepEqual(c.monthDays(month),expected);
}
assert.equal(c.shiftMonth('2026-01',-1),'2025-12');
assert.equal(c.shiftMonth('2026-12',1),'2027-01');
assert.equal(c.shiftMonth('2026-01',-13),'2024-12');
assert.equal(c.monthLabel('2026-10'),'October 2026');
const solo={id:'solo',focus_seconds:600,is_duo:false,private_tag:'work'};
const duo={id:'duo',focus_seconds:1500,is_duo:true,private_tag:'study'};
const other={id:'other',focus_seconds:300,is_duo:true,private_tag:'work'};
const row={day:'2026-10-01',solo_seconds:600,duo_seconds:1500,
           solo_rounds:2,duo_rounds:1,sessions:[solo,duo,other]};
assert.deepEqual(c.dayTotals(row,'all'),{seconds:2100,rounds:3});
assert.deepEqual(c.dayTotals(row,'solo'),{seconds:600,rounds:2});
assert.deepEqual(c.dayTotals(row,'duo'),{seconds:1500,rounds:1});
assert.deepEqual(c.dayTotals(undefined,'all'),{seconds:0,rounds:0});
assert.equal(c.sessionMatches(solo,'all','work'),true);
assert.equal(c.sessionMatches(solo,'duo','work'),false);
assert.equal(c.sessionMatches(duo,'all','work'),false);
assert.equal(c.sessionMatches({...solo,private_tag:null},'solo','untagged'),true);
assert.deepEqual(c.filteredDayTotals(row,'all','all'),{seconds:2100,rounds:3});
assert.deepEqual(c.filteredDayTotals(row,'all','work'),{seconds:900,rounds:2});
assert.deepEqual(c.filteredDayTotals(row,'solo','work'),{seconds:600,rounds:1});
assert.deepEqual(c.filteredDayTotals(undefined,'all','work'),{seconds:0,rounds:0});
for(const [seconds,label] of [[0,'0s'],[59,'59s'],[60,'1m'],[3600,'1h 0m'],[3660,'1h 1m']])
  assert.equal(c.focusLabel(seconds),label);
"""
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            prefix + "const cases = " + json.dumps(cases) + ";" + checks,
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
