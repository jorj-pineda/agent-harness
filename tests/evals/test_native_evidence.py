import json

import httpx
import pytest

from evals import heldout_native_evidence as evidence
from evals.owned_inference_transport import ENDPOINT, OwnedInferenceTransport


@pytest.fixture
def directory(tmp_path):
    return tmp_path.resolve() / "native"


async def test_exact_private_bytes_survive_http_rejection(directory):
    raw = b'{"error":"scripted failure"}'
    transport = evidence.NativeEvidenceTransport(
        httpx.MockTransport(lambda r: httpx.Response(500, content=raw)), directory
    )
    owned = OwnedInferenceTransport(transport, lambda: None)
    try:
        with pytest.raises(ValueError, match="HTTP"):
            await owned.handle_async_request(
                httpx.Request("POST", ENDPOINT, content=b'{"messages":[]}')
            )
        assert (directory / "request-01.request.bin").read_bytes() == b'{"messages":[]}'
        assert (directory / "request-01.response.bin").read_bytes() == raw
        assert json.loads((directory / "request-01.json").read_bytes())["status_code"] == 500
        assert all(p.stat().st_mode & 0o777 == 0o600 for p in directory.iterdir())
    finally:
        await owned.aclose()


@pytest.mark.parametrize("fault", ["request_bound", "write", "credentials", "route", "transport"])
async def test_refusal_and_failure_latch_without_resubmission(directory, monkeypatch, fault):
    calls = []

    def reply(request):
        calls.append(request)
        raise httpx.ReadTimeout("scripted uncertainty")

    transport = evidence.NativeEvidenceTransport(httpx.MockTransport(reply), directory)
    if fault == "request_bound":
        monkeypatch.setattr(evidence, "MAX_BYTES", 2)
    elif fault == "write":
        (directory / "request-01.request.bin").write_bytes(b"preserved")
    request = httpx.Request(
        "POST",
        ENDPOINT if fault != "route" else "https://example.invalid",
        headers={"authorization": "secret-sentinel"} if fault == "credentials" else {},
        content=b"body",
    )
    try:
        with pytest.raises((ValueError, OSError, httpx.ReadTimeout)) as caught:
            await transport.handle_async_request(request)
        with pytest.raises(ValueError):
            await transport.handle_async_request(request)
        assert len(calls) == (1 if fault == "transport" else 0)
        assert "secret-sentinel" not in str(caught.value)
        failure = json.loads((directory / "request-01.failure.json").read_bytes())
        assert failure["delegated"] == (fault == "transport")
        if fault == "write":
            assert (directory / "request-01.request.bin").read_bytes() == b"preserved"
    finally:
        await transport.aclose()


async def test_stream_overflow_records_incomplete_prefix_and_closes(directory, monkeypatch):
    closed = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"a" * 64_000
            yield b"b" * 64_000

        async def aclose(self):
            closed.append(True)

    monkeypatch.setattr(evidence, "MAX_BYTES", 70_000)
    transport = evidence.NativeEvidenceTransport(
        httpx.MockTransport(lambda r: httpx.Response(200, stream=Stream())), directory
    )
    try:
        with pytest.raises(ValueError, match="bound"):
            await transport.handle_async_request(httpx.Request("POST", ENDPOINT, content=b"{}"))
        assert (directory / "request-01.partial-response.bin").read_bytes() == b"a" * 64_000
        assert not (directory / "request-01.response.bin").exists()
        assert not json.loads((directory / "request-01.failure.json").read_bytes())[
            "response_body_complete"
        ]
        assert closed
    finally:
        await transport.aclose()


async def test_success_replays_body_and_request_count_is_bounded(directory, monkeypatch):
    monkeypatch.setattr(evidence, "MAX_REQUESTS", 1)
    transport = evidence.NativeEvidenceTransport(
        httpx.MockTransport(lambda r: httpx.Response(200, content=b"raw")), directory
    )
    try:
        result = await transport.handle_async_request(
            httpx.Request("POST", ENDPOINT, content=b"{}")
        )
        assert result.content == b"raw"
        with pytest.raises(ValueError):
            await transport.handle_async_request(httpx.Request("POST", ENDPOINT, content=b"{}"))
        assert transport.requests == 1
    finally:
        await transport.aclose()


async def test_response_close_failure_retains_bytes_and_latches(directory):
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"payload"

        async def aclose(self):
            raise OSError("scripted-close-failure")

    transport = evidence.NativeEvidenceTransport(
        httpx.MockTransport(lambda r: httpx.Response(200, stream=Stream())), directory
    )
    try:
        with pytest.raises(OSError):
            await transport.handle_async_request(httpx.Request("POST", ENDPOINT, content=b"{}"))
        assert transport.failed
        assert (directory / "request-01.failure.json").exists() or (
            directory / "request-01.close-failure.json"
        ).exists()
        with pytest.raises(ValueError):
            await transport.handle_async_request(httpx.Request("POST", ENDPOINT, content=b"{}"))
    finally:
        await transport.aclose()
