from types import SimpleNamespace

import httpx
import pytest

from evals import heldout_execution_cohort as cohort
from evals import heldout_execution_operator as operator
from evals.heldout_execution_admission import check_admission_withdrawal
from evals.heldout_execution_provider import build_cell_provider
from tests.evals.test_execution_preflight import prepared as prepared
from tests.evals.test_execution_provider import setup as setup


def test_parent_refusal_prevents_transport_and_cohort(tmp_path, monkeypatch):
    calls = []

    def refuse(*args, **kwargs):
        calls.append("admission")
        kwargs["verify_parent"](None, None)

    def parent(*args):
        raise ValueError("no real authority")

    monkeypatch.setattr(operator, "admit_execution", refuse)
    monkeypatch.setattr(operator, "run_admitted_cohort", lambda *a, **k: pytest.fail("cohort"))
    monkeypatch.setattr(
        operator.httpx, "AsyncHTTPTransport", lambda *a, **k: pytest.fail("network")
    )
    with pytest.raises(ValueError, match="authority"):
        operator.execute_selected(
            tmp_path / "manifest.json",
            manifest_sha256="0" * 64,
            approval_path=tmp_path / "approval.json",
            approval_sha256="1" * 64,
            verify_parent=parent,
        )
    assert calls == ["admission"]


def test_selected_operator_orders_admission_and_zero_retry_transport(tmp_path, monkeypatch):
    calls = []
    selected = object()
    result = object()

    def admit(*args, **kwargs):
        calls.append("admitted")
        assert kwargs["manifest_sha256"] == "0" * 64
        return selected

    def network(**kwargs):
        calls.append(("transport", kwargs))
        return "scripted transport fixture; never submits"

    def run(admission, *, transport_factory, scripted):
        assert admission is selected and scripted is False
        assert calls == ["admitted"]
        calls.append("owned-cohort-fixture")
        assert transport_factory(1, tmp_path) == "scripted transport fixture; never submits"
        return result

    monkeypatch.setattr(operator, "admit_execution", admit)
    monkeypatch.setattr(operator, "run_admitted_cohort", run)
    monkeypatch.setattr(operator.httpx, "AsyncHTTPTransport", network)
    assert (
        operator.execute_selected(
            tmp_path / "manifest.json",
            manifest_sha256="0" * 64,
            approval_path=tmp_path / "approval.json",
            approval_sha256="1" * 64,
            verify_parent=lambda *a: None,
        )
        is result
    )
    assert calls[-1] == ("transport", {"retries": 0, "trust_env": False})


async def test_async_entry_refuses_before_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(operator, "admit_execution", lambda *a, **k: pytest.fail("admission"))
    with pytest.raises(ValueError, match="outside an active"):
        operator.execute_selected(
            tmp_path / "m",
            manifest_sha256="0" * 64,
            approval_path=tmp_path / "a",
            approval_sha256="1" * 64,
            verify_parent=lambda *a: None,
        )


def test_real_cohort_refuses_missing_source_provenance_before_other_access():
    with pytest.raises(ValueError, match="provenance"):
        cohort.run_admitted_cohort(
            SimpleNamespace(), transport_factory=lambda *a: pytest.fail("transport"), scripted=False
        )


@pytest.mark.parametrize("phase", ["before", "after", "source_drift"])
async def test_withdrawal_after_admission_blocks_response_and_latches(setup, phase):
    admission, settings, messages, kwargs = setup
    admission.source_manifest = admission.directory.parent / "selected-manifest.json"
    admission.source_manifest.write_bytes((admission.directory / "manifest.json").read_bytes())
    admission.source_manifest.chmod(0o600)
    marker = admission.source_manifest.parent / "WITHDRAWN.json"
    calls = []

    def reply(request):
        calls.append(request)
        marker.write_bytes(b"withdrawn during delegation")
        return httpx.Response(
            200,
            json={
                "model": "gemma4:12b",
                "done": True,
                "message": {"role": "assistant", "content": "scripted"},
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )

    provider = build_cell_provider(
        admission,
        1,
        settings=settings,
        transport=httpx.MockTransport(reply),
        check_owned=lambda: None,
    )
    if phase == "before":
        marker.write_bytes(b"withdrawn after admission")
    elif phase == "source_drift":
        admission.source_manifest.write_bytes(b"changed")
    try:
        with pytest.raises(ValueError):
            await provider.chat(messages, **kwargs)
        with pytest.raises(ValueError):
            await provider.chat(messages, **kwargs)
        assert len(calls) == (1 if phase == "after" else 0)
        with pytest.raises(ValueError, match="hash" if phase == "source_drift" else "withdrawn"):
            check_admission_withdrawal(admission)
    finally:
        await provider.aclose()
