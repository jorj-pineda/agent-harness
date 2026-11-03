"""Scripted Git/hash consistency, never actual parent approval or launch."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).parents[2] / "evals/heldout_comparison/operator_20261102.py"


@pytest.fixture(
    params=[
        "operator_20261102.py",
        "operator_fixes_dev_20261102.py",
        "operator_residency_20261103.py",
        "operator_residency_300_feasibility_20261103.py",
    ]
)
def recipe(request):
    spec = importlib.util.spec_from_file_location(
        "frozen_operator_recipe_fixture", PATH.with_name(request.param)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bindings(recipe, monkeypatch):
    reviewed, merged = "a" * 40, "b" * 40
    approval = SimpleNamespace(
        manifest_sha256=recipe.MANIFEST_SHA,
        runtime_commit=recipe.RUNTIME,
        freeze_reviewed_head=reviewed,
        freeze_merge_commit=merged,
    )
    spec = {
        "manifest_sha256": recipe.MANIFEST_SHA,
        "runtime_commit": recipe.RUNTIME,
        "operator_recipe_sha256": hashlib.sha256(b"recipe").hexdigest(),
        "g0_sha256": hashlib.sha256(b"gate").hexdigest(),
    }
    state = {"parents": "c" * 40 + " " + reviewed}

    def git(*args):
        if args[:2] == ("show", "-s"):
            return state["parents"]
        if args[0] == "merge-base":
            return ""
        if args[0] == "cat-file":
            return str(len(json.dumps(spec)))
        assert args == ("show", f"{reviewed}:{recipe.SPEC_PATH}")
        return json.dumps(spec)

    monkeypatch.setattr(recipe, "git", git)
    monkeypatch.setattr(
        recipe,
        "_read_regular",
        lambda path, limit: (b"gate" if path.name == "result.json" else b"recipe", 0o600),
    )
    return approval, reviewed, merged, spec, state


def test_exact_consistency_bindings_pass_without_establishing_authority(recipe, bindings):
    approval, head, merge, _spec, _state = bindings
    recipe.verify_freeze(approval, head, merge)


@pytest.mark.parametrize(
    "fault",
    [
        "manifest_sha256",
        "runtime_commit",
        "operator_recipe_sha256",
        "g0_sha256",
        "parents",
        "receipt",
        "invalid_commit",
    ],
)
def test_wrong_freeze_bindings_refuse(recipe, bindings, fault):
    approval, head, merge, spec, state = bindings
    if fault == "parents":
        state["parents"] = "c" * 40 + " " + "d" * 40
    elif fault == "receipt":
        approval.freeze_reviewed_head = "e" * 40
    elif fault == "invalid_commit":
        head = "--not-a-commit"
    else:
        spec[fault] = "0" * 64
    with pytest.raises(ValueError):
        recipe.verify_freeze(approval, head, merge)


def test_check_only_never_calls_execute(recipe, monkeypatch):
    monkeypatch.setattr(recipe, "check_candidate", lambda: None)
    monkeypatch.setattr(recipe, "execute_selected", lambda *a, **k: pytest.fail("launch"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["recipe", "--check-only"],
    )
    recipe.main()


def test_execution_missing_parent_arguments_refuses(recipe, monkeypatch):
    monkeypatch.setattr(recipe, "check_candidate", lambda: None)
    monkeypatch.setattr(recipe, "execute_selected", lambda *a, **k: pytest.fail("launch"))
    monkeypatch.setattr(sys, "argv", ["recipe", "--execute"])
    with pytest.raises(SystemExit) as error:
        recipe.main()
    assert error.value.code == 2


def test_no_mode_has_no_default_launch(recipe, monkeypatch):
    monkeypatch.setattr(recipe, "check_candidate", lambda: pytest.fail("candidate inspected"))
    monkeypatch.setattr(sys, "argv", ["recipe"])
    with pytest.raises(SystemExit) as error:
        recipe.main()
    assert error.value.code == 2
