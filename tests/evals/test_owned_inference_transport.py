import asyncio

import httpx
import pytest

from evals.owned_inference_transport import ENDPOINT, OwnedInferenceTransport


async def test_owned_submission_is_checked_before_and_after_without_retry():
    events = []

    def check():
        events.append("ownership")

    def reply(request):
        events.append("delegate")
        return httpx.Response(200, json={"scripted": True})

    transport = OwnedInferenceTransport(httpx.MockTransport(reply), check)
    request = httpx.Request("POST", ENDPOINT, json={})
    assert (await transport.handle_async_request(request)).status_code == 200
    assert events == ["ownership", "delegate", "ownership"]
    assert transport.attempted_requests == 1
    await transport.aclose()
    with pytest.raises(ValueError):
        await transport.handle_async_request(request)


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:11447/api/chat",
        "http://127.0.0.1:11434/api/chat",
        "http://example.com/api/chat",
        "http://127.0.0.1:11447/api/chat?x=1",
        "http://user:password@127.0.0.1:11447/api/chat",
    ],
)
async def test_other_origins_and_queries_refuse_before_delegate(url):
    transport = OwnedInferenceTransport(
        httpx.MockTransport(lambda req: pytest.fail("delegate")), lambda: pytest.fail("ownership")
    )
    with pytest.raises(ValueError):
        await transport.handle_async_request(httpx.Request("POST", url))
    assert transport.failed and transport.attempted_requests == 0
    await transport.aclose()


@pytest.mark.parametrize(
    "fault", ["ownership", "transport", "cancel", "redirect", "status", "after"]
)
async def test_unknown_or_failed_submission_latches_without_resubmission(fault):
    checks = []
    calls = []

    def check():
        checks.append(True)
        if fault == "ownership" or (fault == "after" and len(checks) == 2):
            raise ValueError("unknown ownership")

    def reply(request):
        calls.append(True)
        if fault == "transport":
            raise httpx.ReadTimeout("scripted")
        if fault == "cancel":
            raise asyncio.CancelledError()
        return httpx.Response(302 if fault == "redirect" else 500 if fault == "status" else 200)

    transport = OwnedInferenceTransport(httpx.MockTransport(reply), check)
    request = httpx.Request("POST", ENDPOINT)
    with pytest.raises((ValueError, httpx.HTTPError, asyncio.CancelledError)):
        await transport.handle_async_request(request)
    with pytest.raises(ValueError):
        await transport.handle_async_request(request)
    assert len(calls) == (0 if fault == "ownership" else 1)
    await transport.aclose()


async def test_concurrent_submission_invalidates_original_without_second_delegate():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def reply(request):
        calls.append(True)
        entered.set()
        await release.wait()
        return httpx.Response(200)

    transport = OwnedInferenceTransport(httpx.MockTransport(reply), lambda: None)
    request = httpx.Request("POST", ENDPOINT)
    first = asyncio.create_task(transport.handle_async_request(request))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        with pytest.raises(ValueError):
            await transport.handle_async_request(request)
        release.set()
        with pytest.raises(ValueError):
            await first
        assert calls == [True]
    finally:
        release.set()
        await transport.aclose()
