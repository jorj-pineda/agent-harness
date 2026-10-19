"""Safe provider failure observations; no network or model inference."""

from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from harness.runs import RunInput, RunRecord
from providers.base import ChatMessage, ProviderError, ProviderFailure
from providers.ollama import OllamaProvider


@pytest.mark.parametrize(
    ("exception", "kind"),
    [
        (httpx.ReadTimeout, "read_timeout"),
        (httpx.ConnectTimeout, "connect_timeout"),
        (httpx.WriteTimeout, "write_timeout"),
        (httpx.PoolTimeout, "pool_timeout"),
        (httpx.TimeoutException, "timeout"),
        (httpx.ConnectError, "connection"),
        (httpx.ReadError, "connection"),
        (httpx.RemoteProtocolError, "unknown"),
    ],
)
async def test_ollama_transport_failure_omits_private_exception_text(exception, kind):
    calls = []

    def fail(request):
        calls.append(request)
        raise exception("private-token private-source", request=request)

    provider = OllamaProvider(
        host="http://private-endpoint",
        model="local",
        embed_model="unused",
        transport=httpx.MockTransport(fail),
    )
    try:
        with pytest.raises(ProviderError) as failed:
            await provider.chat([ChatMessage(role="user", content="private-message")])
        assert failed.value.failure.model_dump() == {"kind": kind, "http_status": None}
        assert "private" not in failed.value.failure.model_dump_json()
        assert isinstance(failed.value.__cause__, exception)
        assert len(calls) == 1  # No new provider retry.
    finally:
        await provider.aclose()


@pytest.mark.parametrize("status", [401, 429, 503])
async def test_ollama_http_failure_retains_only_status_not_body_or_url(status):
    def fail(request):
        return httpx.Response(status, text="private-response-body", request=request)

    provider = OllamaProvider(
        host="http://private-endpoint",
        model="local",
        embed_model="unused",
        transport=httpx.MockTransport(fail),
    )
    try:
        with pytest.raises(ProviderError) as failed:
            await provider.chat([ChatMessage(role="user", content="private-message")])
        assert failed.value.failure.model_dump() == {"kind": "http_status", "http_status": status}
        assert "private" not in failed.value.failure.model_dump_json()
    finally:
        await provider.aclose()


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "private-exception"},
        {"message": "secret"},
        {"http_status": True},
        {"http_status": "503"},
        {"http_status": 999},
    ],
)
def test_failure_metadata_rejects_unallowlisted_values(payload):
    with pytest.raises(ValidationError):
        ProviderFailure.model_validate(payload)


def test_unknown_provider_error_and_legacy_run_remain_explicit():
    assert ProviderError("private-error").failure == ProviderFailure(kind="unknown")
    record = RunRecord(
        request=RunInput(user_id="dev", session_id="s", message="x", request_id="k"),
        provider="local",
    )
    legacy = record.model_dump(mode="json")
    legacy.pop("provider_failure")
    assert RunRecord.model_validate(legacy).provider_failure is None
