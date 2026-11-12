"""Offline fresh-development task controls, using the shared external grader."""

from pathlib import Path

import pytest

from evals.real_run import (
    _reference_files,
    _visible_check,
    acceptance_check,
    load_tasks,
    validate_task,
)
from workspace import disposable_workspace

ROOT = Path(__file__).resolve().parents[2] / "evals/fresh_visible_tasks/v1"
TASKS = load_tasks(ROOT / "tasks.yaml")
PARTIAL = {
    "fv_lookup_option": (
        "lookup.py",
        "def option(options, key, default):\n    return options.get(key) or default\n",
    ),
    "fv_merge_windows": (
        "windows.py",
        (ROOT / "reference/fv_merge_windows/windows.py")
        .read_text()
        .replace("start <= merged", "start < merged"),
    ),
    "fv_header_pair": (
        "headers.py",
        (ROOT / "reference/fv_header_pair/headers.py")
        .read_text()
        .replace('text.split(":", 1)', 'text.split(":")'),
    ),
}


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.id)
async def test_initial_rejected_and_reference_passes_both_visible_arms(task):
    await validate_task(task)
    with disposable_workspace(task.fixture) as copied:
        _reference_files(task, copied.workspace.root)
        (copied.workspace.root / "test_visible.py").write_bytes(
            (ROOT / "stronger" / task.id / "test_visible.py").read_bytes()
        )
        assert (await _visible_check(copied.workspace.root)).exit_code == 0
        assert (await acceptance_check(task, copied.workspace.root)).exit_code == 0
    assert task.task_set.partition == "development"


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.id)
async def test_partial_patch_passes_weak_but_stronger_and_external_checks_reject(task):
    before = {
        p: p.read_bytes() for p in ROOT.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml"}
    }
    with disposable_workspace(task.fixture) as copied:
        name, partial = PARTIAL[task.id]
        (copied.workspace.root / name).write_text(partial)
        assert (await _visible_check(copied.workspace.root)).exit_code == 0
        (copied.workspace.root / "test_visible.py").write_bytes(
            (ROOT / "stronger" / task.id / "test_visible.py").read_bytes()
        )
        assert (await _visible_check(copied.workspace.root)).exit_code != 0
        assert (await acceptance_check(task, copied.workspace.root)).exit_code != 0
        # An editable visible test never defines authoritative task acceptance.
        (copied.workspace.root / "test_visible.py").write_text("def test_weakened():\n    pass\n")
        assert (await _visible_check(copied.workspace.root)).exit_code == 0
        assert (await acceptance_check(task, copied.workspace.root)).exit_code != 0
    assert {p: p.read_bytes() for p in before} == before


def test_explicit_development_catalog_does_not_expand_default_tasks():
    assert len(TASKS) == 3
    assert {task.category for task in TASKS} == {"bugfix", "feature", "refactor"}
    assert not {task.id for task in TASKS} & {task.id for task in load_tasks()}
    for task in TASKS:
        assert not task.acceptance.is_relative_to(task.fixture)
        assert not task.reference.is_relative_to(task.fixture)
        assert (ROOT / "stronger" / task.id / "test_visible.py").is_file()


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.id)
def test_stronger_checks_preserve_every_original_visible_test(task):
    import ast

    def functions(path):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef)
        }

    weak = functions(task.fixture / "test_visible.py")
    stronger = functions(ROOT / "stronger" / task.id / "test_visible.py")
    assert weak and all(stronger.get(name) == body for name, body in weak.items())
    assert len(stronger) > len(weak)
