import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from evals import residency_observation as observation
from evals.heldout_comparison import validate_plan


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
