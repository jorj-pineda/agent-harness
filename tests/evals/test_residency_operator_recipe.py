import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(
    params=[
        "operator_residency_20261103.py",
        "operator_residency_300_feasibility_20261103.py",
        "operator_direct_feasibility_20261104.py",
        "operator_no_plan_feasibility_20261104.py",
        "operator_fixed_check_feasibility_20261105.py",
        "operator_whole_file_check_feasibility_20261105.py",
    ]
)
def recipe(request):
    path = Path(__file__).parents[2] / "evals/heldout_comparison" / request.param
    spec = importlib.util.spec_from_file_location("residency_recipe_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        "run_residency_feasibility"
        if hasattr(recipe, "run_residency_feasibility")
        else "run_residency_cohort",
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
        "run_residency_feasibility"
        if hasattr(recipe, "run_residency_feasibility")
        else "run_residency_cohort",
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
