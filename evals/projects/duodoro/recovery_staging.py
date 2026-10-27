"""Opt-in paired-trial staging, never an application restore/resume facility.

Only a newly created owned root may be replaced. The caller must check that its
owned API, tools and listener have settled; this module cannot discover detached
workers or exclude external writers. Ignored contents are neither copied nor
certified. All retained source/grade/store data is private plaintext.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from evals.real_run import Task, acceptance_check
from harness.checkpoints import MAX_CHECKPOINT_ENTRIES, CheckpointFile, capture_checkpoint
from harness.workspace_review import REVIEW_IGNORE

MAX_MANIFEST_BYTES = 1_000_000
MAX_STORE_BYTES = 64_000_000
SettledCheck = Callable[[], None]


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(path: Path) -> Path:
    if path != path.absolute() or path.resolve(strict=True) != path:
        raise ValueError("Staging paths must be absolute and canonical.")
    return path


def _private_write(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write(raw)


def _new_directory(path: Path) -> None:
    _canonical(path.parent)
    path.mkdir(mode=0o700, exist_ok=False)


def _read_regular(path: Path, limit: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("Evidence must be a bounded regular file.")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    named = path.lstat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_mode) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_mode,
    ) or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino):
        raise ValueError("Evidence changed while being read.")
    if len(raw) != before.st_size or len(raw) > limit:
        raise ValueError("Evidence size changed or exceeded its limit.")
    return raw


class TreeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root_identity: tuple[int, int]
    files: dict[str, CheckpointFile] = Field(max_length=MAX_CHECKPOINT_ENTRIES)
    directories: dict[str, int] = Field(max_length=MAX_CHECKPOINT_ENTRIES)
    # lstat metadata at each excluded boundary, not excluded content hashes.
    ignored: dict[str, tuple[int, int, int, int, int, int]] = Field(
        max_length=MAX_CHECKPOINT_ENTRIES
    )


def _capture(root: Path) -> tuple[TreeManifest, dict[str, bytes]]:
    captured = capture_checkpoint(root, "eval-owned-staging")
    directories = {".": stat.S_IMODE(root.stat().st_mode)}
    ignored = {}

    def on_error(error: OSError) -> None:
        raise error

    for parent, dirs, names in os.walk(root, followlinks=False, onerror=on_error):
        for name in sorted((*dirs, *names)):
            path = Path(parent) / name
            value = path.lstat()
            if not (stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode)):
                raise ValueError("Staging refuses symlinks and special files, even if excluded.")
            relative = path.relative_to(root).as_posix()
            if any(fnmatch(name, pattern) for pattern in REVIEW_IGNORE):
                ignored[relative] = (
                    value.st_dev,
                    value.st_ino,
                    value.st_mode,
                    value.st_size,
                    value.st_mtime_ns,
                    value.st_ctime_ns,
                )
            elif stat.S_ISDIR(value.st_mode):
                directories[relative] = stat.S_IMODE(value.st_mode)
        dirs[:] = [
            name for name in dirs if not any(fnmatch(name, pattern) for pattern in REVIEW_IGNORE)
        ]
        if len(directories) + len(ignored) > MAX_CHECKPOINT_ENTRIES:
            raise ValueError("Staging directory/ignored-entry limit exceeded.")
    # A second eligible scan detects changes between content and directory scans.
    again = capture_checkpoint(root, "eval-owned-staging")
    if (
        captured.record.files != again.record.files
        or captured.record.root_identity != again.record.root_identity
    ):
        raise ValueError("Workspace changed during staging capture.")
    return TreeManifest(
        root_identity=captured.record.root_identity,
        files=captured.record.files,
        directories=directories,
        ignored=ignored,
    ), captured.contents


@dataclass(frozen=True)
class SealedTree:
    directory: Path
    origin: Path
    manifest_sha256: str


def seal_tree(root: Path, destination: Path) -> SealedTree:
    """Retain eligible contents/modes and boundary metadata without changing root."""
    _canonical(root)
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("Evidence and workspace trees must not overlap.")
    manifest, contents = _capture(root)
    _new_directory(destination)
    (destination / "blobs").mkdir(mode=0o700)
    for name, raw in contents.items():
        blob = destination / "blobs" / manifest.files[name].sha256
        if not blob.exists():
            _private_write(blob, raw)
    raw = manifest.model_dump_json().encode()
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("Staging manifest limit exceeded.")
    _private_write(destination / "tree.json", raw)
    if _capture(root)[0] != manifest:
        raise ValueError("Source changed while preserving its tree.")
    return SealedTree(destination, root, _digest(raw))


def _load(seal: SealedTree) -> tuple[TreeManifest, dict[str, bytes]]:
    _canonical(seal.directory)
    raw = _read_regular(seal.directory / "tree.json", MAX_MANIFEST_BYTES)
    if _digest(raw) != seal.manifest_sha256:
        raise ValueError("Sealed manifest changed.")
    manifest = TreeManifest.model_validate_json(raw)
    for name in (*manifest.files, *manifest.directories, *manifest.ignored):
        path = Path(name)
        if (
            (name != "." and (not path.parts or str(path) != name))
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise ValueError("Invalid sealed relative path.")
    contents = {}
    for name, entry in manifest.files.items():
        raw = _read_regular(seal.directory / "blobs" / entry.sha256, entry.size)
        if _digest(raw) != entry.sha256:
            raise ValueError("Sealed contents changed.")
        contents[name] = raw
    return manifest, contents


def _protected_unchanged(protected: tuple[SealedTree, ...]) -> None:
    for seal in protected:
        if _capture(seal.origin)[0] != _load(seal)[0]:
            raise ValueError("Protected source/criteria/reference changed.")


@dataclass(frozen=True)
class ScoredArtifact:
    tree: SealedTree
    grade_sha256: str
    protected: tuple[SealedTree, ...]


@dataclass(frozen=True)
class OwnedWorkspace:
    root: Path
    identity: tuple[int, int]
    owner_sha256: str

    @classmethod
    def create(cls, directory: Path) -> OwnedWorkspace:
        """No adoption of an existing directory, including an empty one."""
        _new_directory(directory)
        root = directory / "workspace"
        root.mkdir(mode=0o700)
        info = root.stat()
        identity = (info.st_dev, info.st_ino)
        raw = json.dumps({"root": str(root), "identity": identity}).encode()
        _private_write(directory / "owner.json", raw)
        return cls(root, identity, _digest(raw))

    def _check(self) -> None:
        _canonical(self.root)
        info = self.root.stat()
        if (info.st_dev, info.st_ino) != self.identity or _digest(
            _read_regular(self.root.parent / "owner.json", 4096)
        ) != self.owner_sha256:
            raise ValueError("Owned workspace identity changed.")

    async def score(
        self,
        task: Task,
        destination: Path,
        *,
        protected: tuple[SealedTree, ...],
        settled: SettledCheck,
    ) -> ScoredArtifact:
        """Retain two independent acceptance outputs, including rejected artifacts.

        Caller supplies protected prepared source, criteria and reference seals.
        Exceptions/timeouts/cancellation create no replacement authorization.
        """
        settled()
        self._check()
        if not {task.fixture, task.acceptance, task.reference}.issubset(
            {seal.origin for seal in protected}
        ):
            raise ValueError("Prepared source, criteria and reference seals are required.")
        _protected_unchanged(protected)
        before = _capture(self.root)[0]
        grades = [asdict(await acceptance_check(task, self.root)) for _ in range(2)]
        settled()
        self._check()
        _protected_unchanged(protected)
        after = _capture(self.root)[0]
        if (before.files, before.directories) != (after.files, after.directories):
            raise ValueError("Grading changed the eligible artifact.")
        if grades[0]["exit_code"] != grades[1]["exit_code"]:
            raise ValueError("Independent grade and regrade disagree.")
        if grades[0]["exit_code"] not in (0, 1):
            raise ValueError("Grader infrastructure failure is not a scored artifact.")
        seal = seal_tree(self.root, destination)
        if _load(seal)[0] != after:
            raise ValueError("Artifact changed between grading and preservation.")
        raw = json.dumps(grades, sort_keys=True).encode()
        _private_write(destination / "grades.json", raw)
        return ScoredArtifact(seal, _digest(raw), protected)

    def stage(
        self,
        approved: SealedTree,
        *,
        last_scored: ScoredArtifact | None,
        protected: tuple[SealedTree, ...],
        settled: SettledCheck,
    ) -> None:
        """Replace only exact last-scored eligible work, preserving the root inode.

        None is valid only for the newly empty root. Ignored entries remain in
        place; a structural collision refuses before any mutation. Filesystem I/O
        failure or concurrent writers can still leave a partial stage: halt, do
        not retry or clean automatically. This is not an atomic restore.
        """
        settled()
        self._check()
        if not protected:
            raise ValueError("Protected source/criteria/reference evidence is required.")
        _protected_unchanged(protected)
        target, contents = _load(approved)
        current = _capture(self.root)[0]
        if last_scored is None:
            if current.files or set(current.directories) != {"."} or current.ignored:
                raise ValueError("Initial staging requires an empty owned root.")
        else:
            _protected_unchanged(last_scored.protected)
            expected, _ = _load(last_scored.tree)
            raw = _read_regular(last_scored.tree.directory / "grades.json", 100_000)
            if _digest(raw) != last_scored.grade_sha256 or current != expected:
                raise ValueError("Last-scored proof or current workspace changed.")
            if last_scored.tree.origin != self.root:
                raise ValueError("Last-scored artifact belongs to a different root.")
        # An excluded entry or its parent cannot be removed/replaced incidentally.
        for ignored in current.ignored:
            parents = Path(ignored).parents
            if any(str(p) not in target.directories for p in parents):
                raise ValueError("Staging would remove an ignored entry's parent.")
            if any(p == Path(ignored) or Path(ignored) in p.parents for p in map(Path, contents)):
                raise ValueError("Staging would overwrite an ignored entry.")
        settled()
        self._check()
        _protected_unchanged(protected)
        if last_scored is not None:
            _protected_unchanged(last_scored.protected)
        _load(approved)
        if _capture(self.root)[0] != current:
            raise ValueError("Workspace changed before staging.")
        for name in current.files:
            (self.root / name).unlink()
        for name in sorted(
            set(current.directories) - set(target.directories), key=len, reverse=True
        ):
            (self.root / name).rmdir()
        for name in sorted(target.directories, key=len):
            if name != ".":
                (self.root / name).mkdir(exist_ok=True)
        for name, raw in contents.items():
            _private_write(self.root / name, raw)
            (self.root / name).chmod(target.files[name].mode)
        for name, mode in target.directories.items():
            if name != ".":
                (self.root / name).chmod(mode)
        self._check()
        staged = _capture(self.root)[0]
        if staged.files != target.files or {
            k: v for k, v in staged.directories.items() if k != "."
        } != {k: v for k, v in target.directories.items() if k != "."}:
            raise ValueError("Staged artifact mismatch; stop without cleanup.")
        if staged.ignored != current.ignored:
            raise ValueError("Ignored boundary metadata changed; stop without cleanup.")
        _protected_unchanged(protected)


def clone_closed_stores(
    source: Path,
    destination: Path,
    *,
    expected: dict[str, str],
    settled: SettledCheck,
) -> dict[str, str]:
    """Clone exactly sessions.sqlite/facts.sqlite, with no evidence/path rewrites.

    Inputs must be closed, non-WAL standalone databases. The caller's settlement
    check is required; absence of a journal alone does not prove connections closed.
    Reuse the returned byte hashes to clone each independent arm from this parent.
    Partial destinations are retained on failure, never used/replaced automatically.
    """
    if set(expected) != {"sessions.sqlite", "facts.sqlite"}:
        raise ValueError("Exactly the frozen session and fact stores are required.")
    settled()
    _canonical(source)
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Store images must not overlap.")
    if {p.name for p in source.iterdir()} != set(expected):
        raise ValueError("Store directory contains unexpected files or journal sidecars.")
    contents = {}
    for name, digest in expected.items():
        path = source / name
        raw = _read_regular(path, MAX_STORE_BYTES)
        if _digest(raw) != digest:
            raise ValueError("Parent store bytes changed.")
        # Refuse WAL before opening SQLite, which could otherwise create sidecars.
        if raw[18:20] != b"\x01\x01":
            raise ValueError("Only standalone rollback-journal SQLite stores are supported.")
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
            if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError("Store integrity check failed.")
            if connection.execute("PRAGMA journal_mode").fetchone() == ("wal",):
                raise ValueError("WAL stores require explicit closed standalone preparation.")
        contents[name] = raw
    settled()
    _new_directory(destination)
    for name, raw in contents.items():
        _private_write(destination / name, raw)
    settled()
    if {p.name for p in source.iterdir()} != set(expected) or any(
        _read_regular(source / name, MAX_STORE_BYTES) != raw for name, raw in contents.items()
    ):
        raise ValueError("Parent stores changed while cloning; stop without cleanup.")
    return dict(expected)
