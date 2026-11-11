"""Scripted recipe contracts, never actual parent approval or launch."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.evals import test_frozen_operator_recipe as contracts


@pytest.fixture
def recipe():
    path = Path(__file__).parents[2] / "evals/public_feedback/operator_20261111.py"
    spec = importlib.util.spec_from_file_location("public_operator_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bindings(recipe, monkeypatch):
    import hashlib

    values = contracts.bindings.__wrapped__(recipe, monkeypatch)
    spec = values[3]
    for field in ("evidence_index_sha256", "native_evidence_index_sha256"):
        spec[field] = hashlib.sha256(field.encode()).hexdigest()

    def selected(path, digest, limit):
        field = (
            "evidence_index_sha256"
            if path.name == "evidence-index.json"
            else "native_evidence_index_sha256"
        )
        assert limit == 2_000_000
        if digest != hashlib.sha256(field.encode()).hexdigest():
            raise ValueError("Index differs from reviewed freeze")
        return field.encode()

    monkeypatch.setattr(recipe, "_selected_bytes", selected)
    return values


test_exact_consistency = (
    contracts.test_exact_consistency_bindings_pass_without_establishing_authority
)
test_wrong_freeze_bindings = contracts.test_wrong_freeze_bindings_refuse
test_check_only = contracts.test_check_only_never_calls_execute
test_missing_parent_arguments = contracts.test_execution_missing_parent_arguments_refuses
test_no_default_mode = contracts.test_no_mode_has_no_default_launch


def test_recipe_admission_precedes_transport_and_real_executor(recipe, monkeypatch, tmp_path):
    events = []
    admission = object()

    def admit(*args, **kwargs):
        events.append("admitted")
        return admission

    def transport(**kwargs):
        assert kwargs == {"retries": 0, "trust_env": False}
        events.append("transport")
        return SimpleNamespace()

    def execute(selected, *, transport_factory, scripted):
        assert selected is admission and scripted is False
        events.append("executor")
        transport_factory(1, tmp_path)
        return "finished"

    monkeypatch.setattr(recipe, "admit_execution", admit)
    monkeypatch.setattr(recipe.httpx, "AsyncHTTPTransport", transport)
    monkeypatch.setattr(
        recipe,
        "run_feedback_cohort",
        execute,
    )
    assert (
        recipe.execute_selected(
            tmp_path / "manifest",
            manifest_sha256="selected",
            approval_path=tmp_path / "receipt",
            approval_sha256="receipt-hash",
            verify_parent=lambda *a: None,
        )
        == "finished"
    )
    assert events == ["admitted", "executor", "transport"]


def test_recipe_refusal_does_not_enter_executor_or_construct_transport(
    recipe, monkeypatch, tmp_path
):
    def refuse(*args, **kwargs):
        raise ValueError("parent refused")

    monkeypatch.setattr(recipe, "admit_execution", refuse)
    monkeypatch.setattr(
        recipe,
        "run_feedback_cohort",
        lambda *a, **k: pytest.fail("executor"),
    )
    monkeypatch.setattr(
        recipe.httpx, "AsyncHTTPTransport", lambda *a, **k: pytest.fail("transport")
    )
    with pytest.raises(ValueError, match="parent refused"):
        recipe.execute_selected(
            tmp_path / "manifest",
            manifest_sha256="selected",
            approval_path=tmp_path / "receipt",
            approval_sha256="receipt-hash",
            verify_parent=lambda *a: None,
        )


async def test_recipe_active_loop_refuses_before_admission(recipe, monkeypatch, tmp_path):
    monkeypatch.setattr(recipe, "admit_execution", lambda *a, **k: pytest.fail("admission"))
    with pytest.raises(ValueError, match="synchronous entry"):
        recipe.execute_selected(
            tmp_path / "manifest",
            manifest_sha256="selected",
            approval_path=tmp_path / "receipt",
            approval_sha256="receipt-hash",
            verify_parent=lambda *a: None,
        )


@pytest.mark.parametrize("field", ["evidence_index_sha256", "native_evidence_index_sha256"])
def test_selected_index_drift_refuses(recipe, bindings, field):
    approval, reviewed, merged, spec, _state = bindings
    spec[field] = "0" * 64
    with pytest.raises(ValueError):
        recipe.verify_freeze(approval, reviewed, merged)
