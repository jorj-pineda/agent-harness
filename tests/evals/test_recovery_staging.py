"""Disposable real-file/owned-grader checks; no user repositories or inference."""

import hashlib
import json
import os
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from evals.projects.duodoro import recovery_staging as staging
from evals.real_run import Task, acceptance_check
from harness.checkpoints import capture_checkpoint
from harness.session_store import SessionStore
from harness.state import Session
from memory import FactStore


def settled():
    # Tests own no API, tools or listener. Live orchestration must supply its checks.
    pass


def metadata(root: Path) -> dict[str, tuple[int, int, int, int, int]]:
    # Reads may update atime; replacement must preserve identity, mode and writes.
    return {
        p.name: (s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        for p in root.iterdir()
        for s in [p.lstat()]
    }


@pytest.fixture
def case(tmp_path):
    fixture = tmp_path / "fixture"
    criteria = tmp_path / "acceptance"
    reference = tmp_path / "reference"
    for directory in (fixture, criteria, reference):
        directory.mkdir()
    (fixture / "code.txt").write_text("broken\n")
    (fixture / "keep.txt").write_text("preserve\n")
    (fixture / "keep.txt").chmod(0o644)
    (fixture / "empty").mkdir(mode=0o750)
    (fixture / "empty").chmod(0o750)
    (fixture / "code.txt").chmod(0o640)
    (criteria / "test_acceptance.py").write_text(
        "from pathlib import Path\n"
        "def test_answer():\n"
        '    assert Path("code.txt").read_text() == "fixed\\n"\n'
    )
    (reference / "code.txt").write_text("fixed\n")
    task = Task("test", "fix the code", fixture, criteria, reference)
    protected = tuple(
        staging.seal_tree(directory, tmp_path / ("sealed-" + directory.name))
        for directory in (fixture, criteria, reference)
    )
    workspace = staging.OwnedWorkspace.create(tmp_path / "owned")
    workspace.stage(protected[0], last_scored=None, protected=protected, settled=settled)
    return workspace, task, protected


async def score(
    case: tuple[staging.OwnedWorkspace, Task, tuple[staging.SealedTree, ...]], path: Path
) -> staging.ScoredArtifact:
    workspace, task, protected = case
    return await workspace.score(task, path, protected=protected, settled=settled)


async def test_two_arms_restore_same_source_and_keep_every_scored_artifact(case, tmp_path):
    workspace, _, protected = case
    identity = workspace.root.stat().st_ino
    source = await score(case, tmp_path / "source-artifact")
    assert [
        g["exit_code"] for g in json.loads((source.tree.directory / "grades.json").read_text())
    ] == [1, 1]
    (workspace.root / "code.txt").write_text("fixed\n")
    (workspace.root / "new").mkdir()
    (workspace.root / "new" / "extra").write_bytes(b"arm-one\x00")
    arm_one = await score(case, tmp_path / "arm-one")
    workspace.stage(source.tree, last_scored=arm_one, protected=protected, settled=settled)
    assert workspace.root.stat().st_ino == identity
    assert (workspace.root / "code.txt").read_text() == "broken\n"
    assert (workspace.root / "code.txt").stat().st_mode & 0o777 == 0o640
    assert not (workspace.root / "new").exists()
    assert (workspace.root / "empty").stat().st_mode & 0o777 == 0o750
    # The accepted sibling patch is retained even after staging the shared source.
    manifest, contents = staging._load(arm_one.tree)
    assert contents["new/extra"] == b"arm-one\x00"
    assert contents["code.txt"] == b"fixed\n"
    assert "new" in manifest.directories
    arm_two = await score(case, tmp_path / "arm-two")
    workspace.stage(protected[0], last_scored=arm_two, protected=protected, settled=settled)
    assert workspace.root.stat().st_ino == identity
    for path in (source.tree.directory, arm_one.tree.directory, arm_two.tree.directory):
        assert path.stat().st_mode & 0o777 == 0o700
        assert (path / "grades.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "drift", ["added", "changed", "deleted", "mode", "directory", "symlink", "fifo", "ignored"]
)
async def test_later_unknown_work_refuses_without_discard(case, tmp_path, drift):
    workspace, _, protected = case
    artifact = await score(case, tmp_path / "artifact")
    root = workspace.root
    if drift == "added":
        (root / "later.txt").write_text("later work")
    elif drift == "changed":
        (root / "keep.txt").write_text("later work")
    elif drift == "deleted":
        (root / "keep.txt").unlink()
    elif drift == "mode":
        (root / "keep.txt").chmod(0o600)
    elif drift == "directory":
        (root / "later-empty").mkdir()
    elif drift == "symlink":
        (root / "link").symlink_to(protected[0].origin / "code.txt")
    elif drift == "fifo":
        os.mkfifo(root / "pipe")
    else:
        (root / ".env.later").write_text("never copy or print")
    before = metadata(root)
    with pytest.raises(ValueError):
        workspace.stage(protected[0], last_scored=artifact, protected=protected, settled=settled)
    assert metadata(root) == before
    assert (artifact.tree.directory / "grades.json").exists()


@pytest.mark.parametrize(
    "change", ["criteria", "reference", "fixture", "blob", "grade", "manifest", "owner", "root"]
)
async def test_changed_preservation_proof_refuses_before_replacement(case, tmp_path, change):
    workspace, task, protected = case
    artifact = await score(case, tmp_path / "artifact")
    if change in {"criteria", "reference", "fixture"}:
        directory = {
            "criteria": task.acceptance,
            "reference": task.reference,
            "fixture": task.fixture,
        }[change]
        next(p for p in directory.iterdir() if p.is_file()).write_text("changed")
    elif change == "blob":
        next((artifact.tree.directory / "blobs").iterdir()).write_bytes(b"changed")
    elif change in {"grade", "manifest"}:
        (
            artifact.tree.directory / {"grade": "grades.json", "manifest": "tree.json"}[change]
        ).write_text("changed")
    elif change == "owner":
        (workspace.root.parent / "owner.json").write_text("changed")
    else:
        workspace.root.rename(workspace.root.parent / "original")
        workspace.root.mkdir()
        (workspace.root / "unrelated").write_text("keep me")
    before = metadata(workspace.root)
    with pytest.raises(ValueError):
        workspace.stage(protected[0], last_scored=artifact, protected=protected, settled=settled)
    assert metadata(workspace.root) == before


async def test_unsettled_and_second_gate_drift_and_missing_grade_refuse(case, tmp_path):
    workspace, _, protected = case
    artifact = await score(case, tmp_path / "artifact")

    def unsettled():
        raise ValueError("owned worker is still active")

    with pytest.raises(ValueError, match="worker"):
        workspace.stage(protected[0], last_scored=artifact, protected=protected, settled=unsettled)
    calls = 0

    def changed_during_gate():
        nonlocal calls
        calls += 1
        if calls == 2:
            (workspace.root / "keep.txt").write_text("external work")

    with pytest.raises(ValueError, match="changed before staging"):
        workspace.stage(
            protected[0], last_scored=artifact, protected=protected, settled=changed_during_gate
        )
    assert (workspace.root / "keep.txt").read_text() == "external work"
    with pytest.raises(ValueError, match="empty"):
        workspace.stage(protected[0], last_scored=None, protected=protected, settled=settled)


def test_cannot_adopt_existing_or_redirected_root(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        staging.OwnedWorkspace.create(existing)
    (tmp_path / "link").symlink_to(existing, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical"):
        staging.OwnedWorkspace.create(tmp_path / "link" / "new")
    assert not (existing / "new").exists()


async def test_grader_mutation_or_disagreement_cannot_authorize_stage(case, tmp_path, monkeypatch):
    workspace, task, protected = case
    original = acceptance_check
    calls = 0

    async def inconsistent(task, root):
        nonlocal calls
        calls += 1
        result = await original(task, root)
        return replace(result, exit_code=0 if calls == 1 else 1)

    monkeypatch.setattr(staging, "acceptance_check", inconsistent)
    with pytest.raises(ValueError, match="disagree"):
        await score(case, tmp_path / "no-proof")
    assert not (tmp_path / "no-proof").exists()
    monkeypatch.setattr(staging, "acceptance_check", original)
    (task.acceptance / "test_acceptance.py").write_text(
        "from pathlib import Path\n"
        "def test_mutation():\n"
        '    Path("keep.txt").write_text("changed by grader")\n'
    )
    # Explicitly seal this separate test criterion before running it.
    changed = staging.seal_tree(task.acceptance, tmp_path / "changed-criterion")
    protected = (protected[0], changed, protected[2])
    with pytest.raises(ValueError, match="Grading changed"):
        await workspace.score(task, tmp_path / "no-proof", protected=protected, settled=settled)
    assert (workspace.root / "keep.txt").read_text() == "changed by grader"


async def test_change_between_grade_and_seal_cannot_authorize_stage(case, tmp_path, monkeypatch):
    workspace, _, _ = case
    original = staging.seal_tree

    def changed(root, destination):
        (root / "keep.txt").write_text("late external edit")
        return original(root, destination)

    monkeypatch.setattr(staging, "seal_tree", changed)
    with pytest.raises(ValueError, match="between grading and preservation"):
        await score(case, tmp_path / "artifact")
    assert not (tmp_path / "artifact" / "grades.json").exists()
    assert (workspace.root / "keep.txt").read_text() == "late external edit"


async def test_last_artifact_protections_survive_change_to_next_task(case, tmp_path):
    workspace, task, _ = case
    artifact = await score(case, tmp_path / "artifact")
    alternate = tmp_path / "next-source"
    alternate.mkdir()
    (alternate / "code.txt").write_text("next task")
    next_seal = staging.seal_tree(alternate, tmp_path / "next-seal")
    (task.acceptance / "test_acceptance.py").write_text("changed prior criterion")
    with pytest.raises(ValueError, match="Protected"):
        workspace.stage(next_seal, last_scored=artifact, protected=(next_seal,), settled=settled)
    assert (workspace.root / "code.txt").read_text() == "broken\n"


async def test_ignored_boundary_survives_and_parent_removal_refuses(case, tmp_path):
    workspace, _, protected = case
    cache = workspace.root / "empty" / "__pycache__"
    cache.mkdir()
    (cache / "ignored").write_bytes(b"runtime content")
    artifact = await score(case, tmp_path / "artifact")
    workspace.stage(protected[0], last_scored=artifact, protected=protected, settled=settled)
    assert (cache / "ignored").read_bytes() == b"runtime content"
    # Removing its parent cannot silently discard even an excluded runtime resource.
    target = tmp_path / "without-parent"
    target.mkdir()
    target_seal = staging.seal_tree(target, tmp_path / "without-parent-seal")
    with pytest.raises(ValueError, match="ignored entry's parent"):
        workspace.stage(target_seal, last_scored=artifact, protected=protected, settled=settled)
    assert (cache / "ignored").read_bytes() == b"runtime content"


def test_closed_store_clones_keep_logical_records_and_independent_bytes(tmp_path):
    source = tmp_path / "stores"
    source.mkdir()
    store = SessionStore(source / "sessions.sqlite")
    session = Session(user_id="owner", workspace_root="/literal/frozen/root")
    store.save(session)
    # Use a separate eligible root; SQLite files themselves are not task evidence.
    saved_root = tmp_path / "saved-root"
    saved_root.mkdir()
    (saved_root / "source.txt").write_bytes(b"exact source checkpoint")
    captured = capture_checkpoint(saved_root, session.session_id)
    store.save_checkpoint(captured)
    store.close()
    with FactStore(source / "facts.sqlite") as facts:
        facts.add("owner", "keep source evidence")
    expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    parent = tmp_path / "parent"
    assert (
        staging.clone_closed_stores(source, parent, expected=expected, settled=settled) == expected
    )
    for arm in ("control", "projected"):
        destination = tmp_path / arm
        staging.clone_closed_stores(parent, destination, expected=expected, settled=settled)
        copy = SessionStore(destination / "sessions.sqlite")
        archive = copy.get(session.session_id)
        assert archive is not None and archive.session == session
        assert copy.get_checkpoint(captured.record.checkpoint_id) == captured.record
        assert copy.checkpoint_content(captured.record, "source.txt") == b"exact source checkpoint"
        copy.save(Session(user_id=arm))
        copy.close()
    assert {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.iterdir()
    } == expected
    assert {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()
    } == expected
    (source / "sessions.sqlite-wal").write_bytes(b"unsettled")
    with pytest.raises(ValueError, match="sidecars"):
        staging.clone_closed_stores(source, tmp_path / "bad", expected=expected, settled=settled)
    assert not (tmp_path / "bad").exists()
    (source / "sessions.sqlite-wal").unlink()
    (source / "facts.sqlite").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        staging.clone_closed_stores(source, tmp_path / "bad", expected=expected, settled=settled)


@pytest.mark.parametrize("failure", ["unsettled", "wal", "corrupt", "symlink"])
def test_store_gate_failures_create_no_arm_and_keep_parent(tmp_path, failure):
    source = tmp_path / "stores"
    source.mkdir()
    for name in ("sessions.sqlite", "facts.sqlite"):
        with sqlite3.connect(source / name) as connection:
            connection.execute("CREATE TABLE evidence (value TEXT)")
            connection.execute("INSERT INTO evidence VALUES ('original')")
            if failure == "wal":
                connection.commit()
                connection.execute("PRAGMA journal_mode=WAL")
        connection.close()
    if failure == "corrupt":
        (source / "facts.sqlite").write_bytes(b"not sqlite")
    if failure == "symlink":
        original = tmp_path / "original"
        (source / "facts.sqlite").rename(original)
        (source / "facts.sqlite").symlink_to(original)
    expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}

    def gate():
        if failure == "unsettled":
            raise ValueError("active owned worker")

    with pytest.raises((ValueError, OSError)):
        staging.clone_closed_stores(source, tmp_path / "arm", expected=expected, settled=gate)
    assert not (tmp_path / "arm").exists()
    assert {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()
    } == expected


@pytest.mark.parametrize("limit", ["bytes", "entries"])
def test_capture_limits_leave_no_partial_seal(tmp_path, monkeypatch, limit):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a").write_bytes(b"unchanged")
    (source / "b").write_bytes(b"unchanged")
    monkeypatch.setattr(
        "harness.checkpoints.MAX_CHECKPOINT_" + {"bytes": "BYTES", "entries": "ENTRIES"}[limit],
        1,
    )
    with pytest.raises(ValueError, match="limit exceeded"):
        staging.seal_tree(source, tmp_path / "seal")
    assert not (tmp_path / "seal").exists()
    assert (source / "a").read_bytes() == b"unchanged"
