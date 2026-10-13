"""Prepare a pinned local Duodoro task for the existing real-task runner, without inference."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from evals.real_run import Task
from workspace import disposable_workspace

ROOT = Path(__file__).parent


def prepare_task(repository: Path, destination: Path) -> Task:
    """Caller owns destination cleanup; never edit or reset the source checkout.

    Use only the user-authorized local Duodoro repository. No cloning, dependency
    installation, provider calls or services happen here. Grading still executes
    trusted repository code on the host; this is not process isolation.
    """
    metadata = json.loads((ROOT / "task.json").read_text())
    source = repository.resolve(strict=True)
    destination = destination.resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Evaluation artifacts must be outside the source checkout")
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="duodoro_archive_") as temporary:
        archive_path = Path(temporary) / "source.tar"
        with archive_path.open("wb") as output:
            result = subprocess.run(
                ["git", "-C", str(source), "archive", metadata["source_revision"]],
                stdout=output,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
            )
        if result.returncode:
            raise ValueError("Pinned Duodoro revision is unavailable locally")
        extracted = Path(temporary) / "source"
        extracted.mkdir()
        with tarfile.open(archive_path) as archive:
            archive.extractall(extracted, filter="data")
        # Shared copy exclusions omit .env and ignored runtime/dependency resources.
        with disposable_workspace(extracted) as copy:
            fixture = destination / "fixture"
            shutil.copytree(copy.workspace.root, fixture)
    for name in ("test_visible.py", "preexisting-user-work.txt"):
        if (fixture / name).exists():
            raise ValueError("Task setup would overwrite a pinned source file")
    shutil.copyfile(ROOT / "visible_duration.py", fixture / "test_visible.py")
    (fixture / "preexisting-user-work.txt").write_text("Preserve this setup-owned work.\n")
    acceptance = destination / "acceptance"
    shutil.copytree(ROOT / "acceptance" / "duration_format", acceptance)
    spec = importlib.util.spec_from_file_location(
        "duodoro_acceptance", acceptance / "test_acceptance.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("Acceptance module unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = module.manifest(fixture)
    (acceptance / "baseline.json").write_text(json.dumps(baseline, indent=2, sort_keys=True))
    # Capture criterion bytes before any model runs; callers verify these afterward.
    protected = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in acceptance.iterdir()
        if path.is_file()
    }
    (destination / "acceptance-sha256.json").write_text(json.dumps(protected, sort_keys=True))
    (destination / "source.json").write_text(json.dumps(metadata, indent=2))
    return Task(
        metadata["id"],
        metadata["request"],
        fixture,
        acceptance,
        ROOT / "reference" / "duration_format",
    )


def acceptance_unchanged(destination: Path) -> bool:
    expected = json.loads((destination / "acceptance-sha256.json").read_text())
    directory = destination / "acceptance"
    actual = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
        if path.is_file()
    }
    return actual == expected
