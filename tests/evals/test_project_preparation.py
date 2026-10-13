import json
import shutil
import subprocess

import pytest

from evals.projects.duodoro import prepare


def repository(tmp_path, monkeypatch, *, collision=False):
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.txt").write_text("pinned\n")
    (source / ".env").write_text("excluded fixture data\n")
    if collision:
        (source / "test_visible.py").write_text("existing test\n")
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=Test Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(argv, cwd=source, check=True, capture_output=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()
    definitions = tmp_path / "definitions"
    shutil.copytree(prepare.ROOT, definitions, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("task.json", "task_timer.json", "task_calendar.json"):
        metadata = json.loads((definitions / name).read_text())
        metadata["source_revision"] = revision
        (definitions / name).write_text(json.dumps(metadata))
    monkeypatch.setattr(prepare, "ROOT", definitions)
    return source


@pytest.mark.parametrize("task_name", ["duration_format", "timer_normalization", "calendar_layout"])
def test_preparation_uses_pinned_commit_preserves_dirty_source_and_excludes_env(
    tmp_path, monkeypatch, task_name
):
    source = repository(tmp_path, monkeypatch)
    (source / "value.txt").write_text("later user edit\n")
    destination = tmp_path / "prepared"
    task = prepare.prepare_task(source, destination, task_name=task_name)
    assert (source / "value.txt").read_text() == "later user edit\n"
    assert (task.fixture / "value.txt").read_text() == "pinned\n"
    assert not (task.fixture / ".env").exists()
    assert not (task.fixture / ".git").exists()
    assert (task.fixture / "preexisting-user-work.txt").exists()
    assert prepare.acceptance_unchanged(destination)
    (task.acceptance / "test_acceptance.py").write_text("changed criterion")
    assert not prepare.acceptance_unchanged(destination)


def test_preparation_rejects_artifact_overlap_and_existing_destinations(tmp_path, monkeypatch):
    source = repository(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="outside"):
        prepare.prepare_task(source, source / "evaluation")
    assert not (source / "evaluation").exists()
    destination = tmp_path / "prepared"
    destination.mkdir()
    (destination / "user.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        prepare.prepare_task(source, destination)
    assert (destination / "user.txt").read_text() == "preserve"


def test_preparation_does_not_overwrite_existing_pinned_setup_names(tmp_path, monkeypatch):
    source = repository(tmp_path, monkeypatch, collision=True)
    with pytest.raises(ValueError, match="overwrite"):
        prepare.prepare_task(source, tmp_path / "prepared")
    assert (source / "test_visible.py").read_text() == "existing test\n"


def test_unknown_task_rejected_before_artifact_creation(tmp_path, monkeypatch):
    source = repository(tmp_path, monkeypatch)
    destination = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Unknown frozen"):
        prepare.prepare_task(source, destination, task_name="../other")
    assert not destination.exists()
