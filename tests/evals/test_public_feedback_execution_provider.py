import json
from pathlib import Path

import httpx
import pytest

import evals.public_feedback_execution_provider as factory
from evals.heldout_inputs import _save
from evals.public_envelope import public_settings
from evals.public_feedback_execution_admission import admit_feedback_execution
from providers.base import ChatMessage, ToolSpec
from tests.evals import test_public_feedback_execution as fixtures

captured = fixtures.captured
template = fixtures.template
source = fixtures.source
prepared = fixtures.prepared


@pytest.fixture
def setup(prepared, monkeypatch):
    monkeypatch.setattr(factory, "_runtime_matches", fixtures.gate._runtime_matches)
    root = Path(prepared.output_directory).parent
    mp = root / "manifest.json"
    mh = _save(mp, prepared.model_dump_json().encode())
    ap = root / "approval.json"
    ah = _save(
        ap,
        json.dumps(
            {
                "format": "public-feedback-parent-approval-v1",
                "approval_id": "scripted-only",
                "manifest_sha256": mh,
                "runtime_commit": prepared.runtime_commit,
                "freeze_reviewed_head": "b" * 40,
                "freeze_merge_commit": "c" * 40,
                "exact_head_review_passed": True,
                "ci_passed": True,
                "merged": True,
                "inference_authorized": True,
            }
        ).encode(),
    )
    admission = admit_feedback_execution(
        mp, manifest_sha256=mh, approval_path=ap, approval_sha256=ah, verify_parent=lambda *a: None
    )
    capture = Path(prepared.archive_directory) / "captured"
    body = json.loads((capture / "01-normalized.json").read_bytes())
    wire = json.loads((capture / "native-01/request-01.request.bin").read_bytes())
    messages = [ChatMessage.model_validate(row) for row in body["messages"]]
    kwargs = {
        "tools": [
            ToolSpec(
                name=row["function"]["name"],
                description=row["function"]["description"],
                parameters_schema=row["function"]["parameters"],
            )
            for row in wire["tools"]
        ],
        "temperature": body["temperature"],
        "max_tokens": body["max_tokens"],
    }
    return admission, public_settings(), messages, kwargs


@pytest.mark.parametrize(
    "fault", ["none", "ownership", "runtime", "dependency", "settings", "manifest"]
)
async def test_guarded_native_request_and_predelegation_drift(setup, monkeypatch, fault):
    admission, settings, messages, kwargs = setup
    calls = []
    owned = [True]

    def check_owned():
        if not owned[0]:
            raise ValueError("ownership lost")

    def respond(request):
        calls.append(request)
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

    provider = factory.build_feedback_cell_provider(
        admission,
        1,
        settings=settings,
        transport=httpx.MockTransport(respond),
        check_owned=check_owned,
    )
    if fault == "ownership":
        owned[0] = False
    elif fault == "runtime":
        monkeypatch.setattr(
            factory, "_runtime_matches", lambda *a: (_ for _ in ()).throw(ValueError("drift"))
        )
    elif fault == "dependency":
        monkeypatch.setattr(
            factory,
            "check_dependency_selection",
            lambda *a: (_ for _ in ()).throw(ValueError("drift")),
        )
    elif fault == "settings":
        settings.max_tool_iterations += 1
    elif fault == "manifest":
        (admission.directory / "manifest.json").write_bytes(b"changed")
    try:
        if fault == "none":
            response = await provider.chat(messages, **kwargs)
            assert response.content == "scripted" and len(calls) == 1
            assert (
                calls[0].content
                == (
                    (Path(admission.manifest.archive_directory) / "captured")
                    / "native-01/request-01.request.bin"
                ).read_bytes()
            )
        else:
            with pytest.raises(ValueError):
                await provider.chat(messages, **kwargs)
            with pytest.raises(ValueError):
                await provider.chat(messages, **kwargs)
            assert not calls
    finally:
        await provider.aclose()


@pytest.mark.parametrize("status", [429, 500])
async def test_owned_http_failure_survives_provider_guard_composition(setup, status):
    admission, settings, messages, kwargs = setup
    calls = []

    def reply(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "scripted backend failure"})

    provider = factory.build_feedback_cell_provider(
        admission,
        1,
        settings=settings,
        transport=httpx.MockTransport(reply),
        check_owned=lambda: None,
    )
    try:
        for _ in range(2):
            with pytest.raises(ValueError, match="http_status"):
                await provider.chat(messages, **kwargs)
        assert provider.stop_reason == "http_status" and len(calls) == 1
    finally:
        await provider.aclose()


@pytest.mark.parametrize("phase", ["before", "after"])
async def test_withdrawal_at_owned_boundary_refuses_and_latches(setup, phase):
    admission, settings, messages, kwargs = setup
    calls = []
    marker = admission.source_manifest.parent / "WITHDRAWN.json"

    def response(request):
        calls.append(request)
        if phase == "after":
            marker.write_text("withdrawn")
        return httpx.Response(
            200,
            json={
                "model": "gemma4:12b",
                "done": True,
                "message": {"role": "assistant", "content": "must not reach runtime"},
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )

    provider = factory.build_feedback_cell_provider(
        admission,
        1,
        settings=settings,
        transport=httpx.MockTransport(response),
        check_owned=lambda: None,
    )
    if phase == "before":
        marker.write_text("withdrawn")
    try:
        with pytest.raises(ValueError):
            await provider.chat(messages, **kwargs)
        with pytest.raises(ValueError):
            await provider.chat(messages, **kwargs)
        assert len(calls) == (phase == "after")
    finally:
        await provider.aclose()


@pytest.mark.parametrize("fault", ["wrong_model", "missing_usage", "not_done"])
async def test_invalid_native_tool_response_never_reaches_caller(setup, fault):
    admission, settings, messages, kwargs = setup
    calls = []

    def reply(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "model": "other" if fault == "wrong_model" else "gemma4:12b",
                "done": fault != "not_done",
                "prompt_eval_count": None if fault == "missing_usage" else 10,
                "eval_count": 5,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "write_file",
                                "arguments": {
                                    "path": "pagination.py",
                                    "content": "must not be dispatched",
                                },
                            }
                        }
                    ],
                },
            },
        )

    provider = factory.build_feedback_cell_provider(
        admission,
        1,
        settings=settings,
        transport=httpx.MockTransport(reply),
        check_owned=lambda: None,
    )
    try:
        for _ in range(2):
            with pytest.raises(ValueError):
                await provider.chat(messages, **kwargs)
        assert len(calls) == 1 and provider.stop_reason
    finally:
        await provider.aclose()


@pytest.mark.parametrize("fault", ["settings", "ownership"])
async def test_post_delegation_drift_refuses_response_and_latches(setup, fault):
    admission, settings, messages, kwargs = setup
    owned = [True]
    calls = []

    def check_owned():
        if not owned[0]:
            raise ValueError("ownership lost")

    def reply(request):
        calls.append(request)
        if fault == "settings":
            settings.max_tool_iterations += 1
        else:
            owned[0] = False
        return httpx.Response(
            200,
            json={
                "model": "gemma4:12b",
                "done": True,
                "message": {"role": "assistant", "content": "must not be returned"},
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )

    provider = factory.build_feedback_cell_provider(
        admission,
        1,
        settings=settings,
        transport=httpx.MockTransport(reply),
        check_owned=check_owned,
    )
    try:
        for _ in range(2):
            with pytest.raises(ValueError):
                await provider.chat(messages, **kwargs)
        assert len(calls) == 1
    finally:
        await provider.aclose()
