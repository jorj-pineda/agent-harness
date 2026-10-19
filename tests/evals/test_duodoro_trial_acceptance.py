"""Offline criterion tests on small pinned library snapshots, never the user's repo."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from evals.projects.duodoro import prepare
from evals.real_run import (
    _reference_files,
    _visible_check,
    acceptance_check,
    run_task,
    validate_task,
)
from harness.config import Settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from workspace import disposable_workspace

FIXTURE = Path(__file__).parents[1] / "fixtures" / "duodoro_trial"
NEW_TASKS = tuple(prepare.TASK_DEFINITIONS)[3:]


@pytest.fixture
def repository(tmp_path, monkeypatch):
    source = tmp_path / "source"
    shutil.copytree(FIXTURE, source)
    provenance = json.loads((source / "source.json").read_text())
    for name, digest in provenance["files_sha256"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(argv, cwd=source, check=True, capture_output=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    definitions = tmp_path / "definitions"
    shutil.copytree(prepare.ROOT, definitions, ignore=shutil.ignore_patterns("__pycache__"))
    for name, _ in prepare.TASK_DEFINITIONS.values():
        path = definitions / name
        metadata = json.loads(path.read_text())
        metadata["source_revision"] = revision
        path.write_text(json.dumps(metadata))
    monkeypatch.setattr(prepare, "ROOT", definitions)
    (source / "user-untracked.txt").write_text("later user work\n")
    return source


def task_for(repository, tmp_path, name):
    return prepare.prepare_task(repository, tmp_path / ("prepared-" + name), task_name=name)


@pytest.mark.parametrize("name", prepare.TASK_DEFINITIONS)
async def test_pinned_library_baseline_fails_reference_passes_without_touching_source(
    repository, tmp_path, name
):
    task = task_for(repository, tmp_path, name)
    await validate_task(task)
    assert prepare.acceptance_unchanged(task.fixture.parent)
    assert (task.fixture / "visible_check.mjs").is_file() is (name in NEW_TASKS)
    assert not (task.fixture / "user-untracked.txt").exists()
    assert (repository / "user-untracked.txt").read_text() == "later user work\n"
    provenance = json.loads((FIXTURE / "source.json").read_text())
    for path, digest in provenance["files_sha256"].items():
        assert hashlib.sha256((repository / path).read_bytes()).hexdigest() == digest


def incorrect_patch(name, root):
    edits = {
        "share_storage": (
            "client/src/lib/shareInvite.ts",
            "try { sessionStorage.removeItem(PENDING_SHARE_INVITE_KEY); } catch { /* unavailable */ }",
            "sessionStorage.removeItem(PENDING_SHARE_INVITE_KEY);",
        ),
        "timer_prefs": (
            "client/src/lib/timerPrefs.ts",
            "...readTimerPrefs(), ...prefs",
            "...DEFAULT_TIMER_PREFS, ...prefs",
        ),
        "shortcut_count": (
            "client/src/lib/keyboardShortcuts.ts",
            "Number.isSafeInteger",
            "Number.isFinite",
        ),
        "duration_refactor": (
            "server/payloadParsers.js",
            "breakDuration: boundedDuration(payload.breakDuration, 5 * 60, 30, MAX_BREAK)",
            "breakDuration: Math.floor(boundedDuration(payload.breakDuration, 5 * 60, 30, MAX_BREAK))",
        ),
        "tag_filter": (
            "client/src/lib/focusTags.ts",
            "return isTagFilter(value) ? value : 'all';",
            "return typeof value === 'string' && isTagFilter(value.trim().toLowerCase()) ? value.trim().toLowerCase() : 'all';",
        ),
        "wrapup_intentions": (
            "client/src/lib/sessionWrapUp.ts",
            "text.trim()",
            "text.trim().toLowerCase()",
        ),
    }
    if name == "rotation_preview":
        # Visible checks inspect only the client: an absent server helper must fail externally.
        path = "server/rotation.js"
        (root / path).write_bytes((FIXTURE / path).read_bytes())
    else:
        path, before, after = edits[name]
        target = root / path
        text = target.read_text()
        assert before in text
        target.write_text(text.replace(before, after, 1))


@pytest.mark.parametrize("name", NEW_TASKS)
async def test_incorrect_patch_can_pass_visible_check_but_external_acceptance_rejects(
    repository, tmp_path, name
):
    task = task_for(repository, tmp_path, name)
    with disposable_workspace(task.fixture) as copy:
        _reference_files(task, copy.workspace.root)
        incorrect_patch(name, copy.workspace.root)
        visible = await _visible_check(copy.workspace.root)
        assert visible.exit_code == 0, visible.stdout + visible.stderr
        external = await acceptance_check(task, copy.workspace.root)
        assert external.exit_code == 1, external.stdout + external.stderr
        assert "test_behavior_and_preserved_exports" in external.stdout
    assert prepare.acceptance_unchanged(task.fixture.parent)


@pytest.mark.parametrize("change", ["visible_python", "visible_node", "sentinel", "extra", "mode"])
async def test_scope_rejects_test_weakening_unrelated_changes_and_mode_changes(
    repository, tmp_path, change
):
    task = task_for(repository, tmp_path, "tag_filter")
    with disposable_workspace(task.fixture) as copy:
        root = copy.workspace.root
        _reference_files(task, root)
        if change == "mode":
            target = root / "client/src/lib/focusTags.ts"
            target.chmod(target.stat().st_mode ^ 0o100)
        else:
            path = {
                "visible_python": "test_visible.py",
                "visible_node": "visible_check.mjs",
                "sentinel": "preexisting-user-work.txt",
                "extra": "unrelated.txt",
            }[change]
            (root / path).write_text(
                "# weakened or unrelated\n" if change == "visible_python" else "\n"
            )
        external = await acceptance_check(task, root)
        assert external.exit_code == 1, external.stdout + external.stderr
        assert "test_scope_and_preexisting_work" in external.stdout


@pytest.mark.parametrize("path", ["checks.mjs", "golden.json", "baseline.json"])
def test_hash_guard_covers_every_external_criterion_input(repository, tmp_path, path):
    task = task_for(repository, tmp_path, "rotation_preview")
    assert prepare.acceptance_unchanged(task.fixture.parent)
    (task.acceptance / path).write_text("changed\n")
    assert not prepare.acceptance_unchanged(task.fixture.parent)


async def test_unused_refactor_export_is_not_full_acceptance_without_independent_review(
    repository, tmp_path
):
    task = task_for(repository, tmp_path, "duration_refactor")
    with disposable_workspace(task.fixture) as copy:
        path = copy.workspace.root / "server/payloadParsers.js"
        path.write_text(
            path.read_text()
            + "\nmodule.exports.boundedDuration = (value,fallback,min,max) => Math.min(Math.max(Number(value)||fallback,min),max);\n"
        )
        # Behavioral equivalence cannot prove reuse. The trial's source-review gate stays pending.
        assert (await acceptance_check(task, copy.workspace.root)).exit_code == 0
    metadata = json.loads((task.fixture.parent / "source.json").read_text())
    assert metadata["requires_independent_source_review"] is True
    row = next(
        row
        for row in json.loads((prepare.ROOT / "trial-ledger.json").read_text())["tasks"]
        if row["id"] == "T08"
    )
    assert row["requires_independent_refactor_source_review"] is True
    assert row["independent_source_review"] is None and row["artifact_acceptance"] is None


@pytest.mark.parametrize("correct", [False, True])
async def test_shared_runtime_scripted_patch_keeps_visible_pass_separate_from_acceptance(
    repository, tmp_path, correct
):
    task = task_for(repository, tmp_path, "timer_prefs")
    content = (task.reference / "client/src/lib/timerPrefs.ts").read_text()
    if not correct:
        content = content.replace(
            "...readTimerPrefs(), ...prefs", "...DEFAULT_TIMER_PREFS, ...prefs"
        )
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={"path": "client/src/lib/timerPrefs.ts", "content": content},
                )
            ]
        ),
        make_response(
            tool_calls=[
                ToolCall(
                    id="check",
                    name="run_command",
                    arguments={"argv": ["python", "-m", "pytest", "-q", "test_visible.py"]},
                )
            ]
        ),
        make_response(content="Check passed; inspect independent acceptance."),
    )
    result = await run_task(
        task,
        provider=provider,
        settings=Settings(
            _env_file=None,
            require_verification_before_finish=True,
            project_check_argv=["python", "-m", "pytest", "-q", "test_visible.py"],
        ),
        mode="harness",
    )
    assert result.verification_status == "passed"
    assert result.completion_status == "completed"
    assert result.passed is correct and result.category == "bugfix"
    assert prepare.acceptance_unchanged(task.fixture.parent)
