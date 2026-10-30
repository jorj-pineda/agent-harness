import json

import httpx
import pytest

from evals import heldout_environment as environment
from evals.heldout_comparison import validate_plan


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:11434",
        "http://example.com",
        "http://localhost:11434",
        "http://127.0.0.1:11434/api/chat",
        "http://user:secret@127.0.0.1:11434",
        "http://127.0.0.1:11434?key=secret",
        "http://127.0.0.1:11434#fragment",
    ],
)
async def test_nonliteral_nonlocal_or_credentialed_origin_refuses_before_transport(endpoint):
    def forbidden(request):
        pytest.fail("request sent")

    with pytest.raises(ValueError):
        await environment.observe(endpoint, transport=httpx.MockTransport(forbidden))


@pytest.mark.parametrize("change", ["none", "version", "digest", "missing"])
async def test_only_metadata_gets_are_sent_and_matches_are_explicit(change):
    plan = validate_plan()
    seen = []

    def respond(request):
        seen.append((request.method, request.url.path))
        assert "authorization" not in request.headers
        if request.url.path == "/api/version":
            return httpx.Response(
                200,
                json={"version": "other" if change == "version" else plan.expected_ollama_version},
            )
        return httpx.Response(
            200,
            json={
                "models": []
                if change == "missing"
                else [
                    {
                        "name": plan.model,
                        "model": plan.model,
                        "digest": "0" * 64 if change == "digest" else plan.model_digest,
                    }
                ]
            },
        )

    result = await environment.observe(
        "http://127.0.0.1:11434", transport=httpx.MockTransport(respond)
    )
    assert seen == [("GET", "/api/version"), ("GET", "/api/tags")]
    assert result["matches_protocol"] is (change == "none")
    assert result["generation_requests"] == 0 and not result["resource_ownership_verified"]
    assert not result["inference_authorized"]


@pytest.mark.parametrize("failure", ["redirect", "oversized", "malformed", "duplicate"])
async def test_invalid_metadata_refuses_without_following_redirects(failure):
    plan = validate_plan()
    calls = []

    def respond(request):
        calls.append(request.url.path)
        if failure == "redirect":
            return httpx.Response(302, headers={"location": "http://public.example/api/chat"})
        if failure == "oversized":
            return httpx.Response(200, content=b"x" * 1000001)
        if failure == "malformed":
            return httpx.Response(200, json={"version": 3})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": plan.expected_ollama_version})
        row = {"name": plan.model, "model": plan.model, "digest": plan.model_digest}
        return httpx.Response(200, json={"models": [row, row]})

    with pytest.raises((ValueError, httpx.HTTPError)):
        await environment.observe("http://127.0.0.1:11434", transport=httpx.MockTransport(respond))
    assert calls[0] == "/api/version" and len(calls) <= 2


async def test_unavailable_environment_is_preserved_without_raw_exception_text(
    monkeypatch, tmp_path
):
    async def failed(*args, **kwargs):
        raise httpx.ConnectError("private exception text")

    monkeypatch.setattr(environment, "observe", failed)
    report = tmp_path / "metadata.json"
    result = await environment.record_observation("http://127.0.0.1:11434", report)
    assert result["status"] == "unavailable" and result["error_type"] == "ConnectError"
    assert "private exception text" not in report.read_text()
    assert json.loads(report.read_text()) == result
    with pytest.raises(FileExistsError):
        await environment.record_observation("http://127.0.0.1:11434", report)
