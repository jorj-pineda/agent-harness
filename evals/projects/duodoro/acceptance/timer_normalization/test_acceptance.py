"""Independent acceptance. This directory and baseline.json stay outside the workspace."""

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

IGNORED = {".git", "node_modules", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
TARGET = "server/payloadParsers.js"


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


def test_integer_timer_contract_and_other_parsers():
    result = subprocess.run(
        [
            "node",
            "-e",
            """
const assert = require('node:assert/strict');
const p = require('./server/payloadParsers.js');
assert.equal(p.MAX_FOCUS, 7200); assert.equal(p.MAX_BREAK, 3600);
const samples = [undefined, null, false, true, '', 'bad', '0', '65.99', 0,
                 -0.1, -999, NaN, Infinity, -Infinity, 25.9, 30.99, 59.99,
                 60, 60.99, 300.99, 1500.9, 3600.9, 7200.9, [], [120.9], {}];
for (const value of samples) {
  for (const mode of ['flow', 'pomodoro', 'FLOW', undefined, null, 42]) {
    const actual = p.parseStartSession({sessionId: 'room', focusDuration: value,
                                       breakDuration: value, mode, unknown: 'ignored'});
    assert.deepEqual(actual, {ok: true, value: {sessionId: 'room',
      focusDuration: Math.floor(Math.min(Math.max(Number(value) || 1500, 60), 7200)),
      breakDuration: Math.floor(Math.min(Math.max(Number(value) || 300, 30), 3600)),
      mode: mode === 'flow' ? 'flow' : 'pomodoro'}});
  }
}
for (const sessionId of [undefined, null, 1, false, {}, []]) {
  assert.deepEqual(p.parseStartSession({sessionId}), {ok: false, reason: 'session_id'});
}
assert.deepEqual(p.parseStartSession({sessionId: ''}), {ok: true, value: {
  sessionId: '', focusDuration: 1500, breakDuration: 300, mode: 'pomodoro'}});
assert.throws(() => p.parseStartSession({sessionId: 'room',
  focusDuration: Symbol('x')}), TypeError);
const avatar = {skinColor: '#F1C27D', hairStyle: 'bob', hairColor: '#3B2314',
                eyeStyle: 'normal', outfitColor: '#4A6FA5'};
assert.deepEqual(p.parseDeleteAccount({confirmation: 'DELETE'}),
                 {ok: true, value: {confirmation: 'DELETE'}});
assert.deepEqual(p.parseDeleteAccount({confirmation: 'delete'}), {ok: false,
  reason: 'confirmation'});
assert.deepEqual(p.parseOnlineFriends({friendIds: ['a','b']}), {ok: true,
  value: {friendIds: ['a','b']}});
for (const friendIds of ['a', [null],
  Array(101).fill('a')]) assert.equal(p.parseOnlineFriends({friendIds}).ok, false);
assert.deepEqual(p.parseSendInvite({targetUserId: 'friend', fromName: 42, sessionId: 42}),
                 {ok: true, value: {targetUserId: 'friend', fromName: 'Someone', sessionId: null}});
assert.equal(p.parseSendInvite({targetUserId: ''}).ok, false);
assert.deepEqual(p.parseCreateSession({avatar, displayName: 'x'.repeat(60), pet: 'griffin'}),
                 {ok: true, value: {avatar, displayName: 'x'.repeat(50), pet: null,
  petName: null}});
for (const value of [null, [], {...avatar, skinColor: 'red'}, {...avatar, hairStyle: 'unknown'}])
  assert.deepEqual(p.parseCreateSession({avatar: value}), {ok: false, reason: 'avatar'});
assert.deepEqual(p.parseJoinSession({sessionId: 'room', avatar}), {ok: true, value: {
  sessionId: 'room', shareToken: null, joiningByLink: false, avatar,
  displayName: 'Player', pet: null, petName: null}});
assert.equal(p.parseJoinSession({shareToken: 'a'.repeat(20), avatar}).value.joiningByLink, true);
assert.deepEqual(p.parseJoinSession({shareToken: 'short', avatar}), {ok: false,
  reason: 'share_token'});
for (const name of ['parseShareInvite','parseSessionReference']) {
  assert.deepEqual(p[name]({sessionId: 'room'}), {ok: true, value: {sessionId: 'room'}});
  assert.deepEqual(p[name]({sessionId: 42}), {ok: false, reason: 'session_id'});
}
assert.deepEqual(p.parseSetPet({sessionId: 'room', pet: 'cat'}), {ok: true, value: {
  sessionId: 'room', pet: 'cat', petName: 'Mochi'}});
assert.deepEqual(p.parseSetPet({sessionId: 'room', pet: 'griffin'}), {ok: true, value: {
  sessionId: 'room', pet: null, petName: null}});
assert.deepEqual(p.parseSetPetName({sessionId: 'room', name: '  Cat  '}),
                 {ok: true, value: {sessionId: 'room', name: 'Cat'}});
assert.deepEqual(p.parseSetIntention({sessionId:'room', text:'  work   well  '}),
                 {ok:true,value:{sessionId:'room',text:'work well'}});
assert.equal(p.parseSetIntention({sessionId:'room',text:'x'.repeat(161)}).ok,false);
assert.deepEqual(p.parseResolveIntention({sessionId:'room',round:1,action:'done'}),
                 {ok:true,value:{sessionId:'room',round:1,action:'done'}});
assert.equal(p.parseResolveIntention({sessionId:'room',round:1.5,action:'done'}).ok,false);
assert.deepEqual(p.parseReaction({sessionId:'room',reaction:'heart'}),
                 {ok:true,value:{sessionId:'room',reaction:'heart'}});
assert.equal(p.parseReaction({sessionId:'room',reaction:'other'}).ok,false);
""",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
