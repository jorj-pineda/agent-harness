import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from evals import residency_observation as observation
from evals.heldout_comparison import validate_plan
from evals.heldout_rehearsal import ProtocolStop, stop_reason


@pytest.mark.parametrize("loaded", [False, True])
async def test_bounded_get_records_point_state_without_generation(tmp_path, monkeypatch, loaded):
    actual = httpx.AsyncClient
    plan = validate_plan()
    requests, owned = [], []

    def respond(request):
        requests.append((request.method, request.url.path))
        return httpx.Response(
            200,
            json={
                "models": [{"name": plan.model, "model": plan.model, "digest": plan.model_digest}]
                if loaded
                else []
            },
        )

    def client(**kwargs):
        assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        return actual(**kwargs, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(observation.httpx, "AsyncClient", client)
    await observation.record_residency(tmp_path, 1, 300, lambda: owned.append("check"))
    assert requests == [("GET", "/api/ps")]
    assert owned == ["check", "check"]
    result = json.loads((tmp_path / "residency-01.json").read_text())
    assert result["state"] == ("loaded" if loaded else "not_loaded")
    assert result["continuous_residency_verified"] is False


async def test_observation_failure_prevents_tool_response_and_latches(tmp_path):
    response = object()
    inner = SimpleNamespace(
        name="fake", stop_reason=None, chat=AsyncMock(return_value=response), aclose=AsyncMock()
    )
    observe = AsyncMock(side_effect=ValueError("unknown state"))
    provider = observation.ResidencyObservedProvider(inner, observe)
    with pytest.raises(ValueError, match="unknown state"):
        await provider.chat([])
    with pytest.raises(ValueError, match="latched closed"):
        await provider.chat([])
    assert inner.chat.await_count == observe.await_count == 1
    assert provider.stop_reason == "residency_observation"
    await provider.aclose()
    inner.aclose.assert_awaited_once()


async def test_observation_completes_before_response_is_returned():
    events = []
    response = object()
    inner = SimpleNamespace(name="fake", stop_reason=None, chat=AsyncMock(return_value=response))

    async def observe(ordinal):
        events.append(ordinal)

    result = await observation.ResidencyObservedProvider(inner, observe).chat([])
    assert result is response and events == [1]


async def test_cancelled_generation_preserves_unknown_usage_reason_and_latches():
    error = asyncio.CancelledError()
    inner = SimpleNamespace(
        name="fake", stop_reason=None, chat=AsyncMock(side_effect=error), aclose=AsyncMock()
    )
    observe = AsyncMock()
    provider = observation.ResidencyObservedProvider(inner, observe)
    with pytest.raises(asyncio.CancelledError) as caught:
        await provider.chat([])
    assert caught.value is error
    observe.assert_not_awaited()
    assert provider.responses == 0
    result = SimpleNamespace(
        termination="budget_exhausted",
        model=validate_plan().model,
        prompt_tokens=None,
        completion_tokens=None,
    )
    assert (provider.stop_reason or stop_reason(result, validate_plan())) == "missing_usage"
    with pytest.raises(ValueError, match="latched closed"):
        await provider.chat([])
    assert inner.chat.await_count == 1
    await provider.aclose()
    inner.aclose.assert_awaited_once()


@pytest.mark.parametrize("reason", ["transport", "wrong_model", "missing_usage"])
async def test_inner_guard_reason_is_preserved_without_observation(reason):
    error = ProtocolStop(reason)
    inner = SimpleNamespace(name="fake", stop_reason=reason, chat=AsyncMock(side_effect=error))
    observe = AsyncMock()
    provider = observation.ResidencyObservedProvider(inner, observe)
    with pytest.raises(ProtocolStop) as caught:
        await provider.chat([])
    assert caught.value is error and provider.stop_reason == reason
    observe.assert_not_awaited()
    with pytest.raises(ValueError, match="latched closed"):
        await provider.chat([])
    assert inner.chat.await_count == 1


async def test_cancelled_observation_retains_observation_reason_and_latches():
    inner = SimpleNamespace(name="fake", stop_reason=None, chat=AsyncMock(return_value=object()))
    observe = AsyncMock(side_effect=asyncio.CancelledError())
    provider = observation.ResidencyObservedProvider(inner, observe)
    with pytest.raises(asyncio.CancelledError):
        await provider.chat([])
    assert provider.stop_reason == "residency_observation" and provider.responses == 1
    with pytest.raises(ValueError, match="latched closed"):
        await provider.chat([])
    assert inner.chat.await_count == observe.await_count == 1


@pytest.mark.parametrize("fault", ["identity", "ownership"])
async def test_invalid_identity_or_ownership_retains_no_success_observation(
    tmp_path, monkeypatch, fault
):
    actual = httpx.AsyncClient
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, json={"models": [{"name": "wrong", "model": "wrong", "digest": "0" * 64}]}
        )

    monkeypatch.setattr(
        observation.httpx,
        "AsyncClient",
        lambda **kwargs: actual(**kwargs, transport=httpx.MockTransport(respond)),
    )

    def owned():
        if fault == "ownership":
            raise ValueError("ownership unknown")

    with pytest.raises(ValueError):
        await observation.record_residency(tmp_path, 1, 0, owned)
    assert not list(tmp_path.iterdir())
    assert len(requests) == (1 if fault == "identity" else 0)
