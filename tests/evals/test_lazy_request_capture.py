"""Delayed diagnostic allocation, not resumable recording or launch permission."""

import json

import httpx
import pytest

from evals.projects.duodoro.provider_probe import FirstRequestGuard
from evals.projects.duodoro.request_capture import (
    CapturedProvider,
    CapturedTransport,
    RequestCapture,
)
from harness.trace import request_identity
from providers.base import ChatMessage
from providers.openai import OpenAIProvider, compatible_transport_options
from tests.evals.test_compatible_request_capture import completed


def setup(directory, respond):
    capture = RequestCapture(
        directory, lazy=True, chat_path="/inference/go/openai/v1/chat/completions"
    )
    adapter = OpenAIProvider(
        api_key="SCRIPTED-private-key",
        model="SCRIPTED-not-Beam",
        provider_name="openai_compatible",
        base_url="https://fixture.invalid/inference/go/openai/v1",
        http_client=httpx.AsyncClient(
            transport=CapturedTransport(httpx.MockTransport(respond), capture)
        ),
        **compatible_transport_options("opencode_go"),
    )
    return CapturedProvider(adapter, capture), capture, adapter


@pytest.mark.parametrize("existing", [False, True])
async def test_construct_bind_and_close_leave_absent_or_existing_inputs_untouched(
    tmp_path, existing
):
    directory = tmp_path / "inputs"
    if existing:
        directory.mkdir()
        (directory / "sentinel").write_bytes(b"retained evidence")
    before = {p.name: p.read_bytes() for p in directory.iterdir()} if existing else None
    provider, _capture, adapter = setup(directory, lambda _: pytest.fail("Unexpected dispatch"))
    provider.for_conversation("session")
    await provider.aclose()
    assert adapter._client.is_closed()
    if existing:
        assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    else:
        assert not directory.exists()


async def test_first_actual_chat_allocates_private_pairs_before_forwarding(tmp_path):
    directory = tmp_path / "inputs"
    seen = []

    def respond(request):
        seen.append(request)
        number = len(seen)
        assert (directory / f"request-{number:04d}-wire.json").read_bytes() == request.content
        assert (
            json.loads((directory / f"request-{number:04d}-normalized.json").read_bytes())[
                "sequence"
            ]
            == number
        )
        assert request.headers["x-opencode-session"] == "session"
        return completed()

    provider, capture, _adapter = setup(directory, respond)
    try:
        view = provider.for_conversation("session")
        assert not directory.exists()
        for _ in range(2):
            await view.chat([ChatMessage(role="user", content="task")], max_tokens=2048)
        assert capture.requests == 2 and len(seen) == 2
        assert directory.stat().st_mode & 0o777 == 0o700
        assert all(p.stat().st_mode & 0o777 == 0o600 for p in directory.iterdir())
    finally:
        await provider.aclose()


async def test_existing_capture_refuses_chat_before_any_forward_or_overwrite(tmp_path):
    directory = tmp_path / "inputs"
    directory.mkdir()
    (directory / "sentinel").write_bytes(b"retained")
    provider, _capture, _adapter = setup(directory, lambda _: pytest.fail("Unexpected dispatch"))
    try:
        view = provider.for_conversation("session")
        with pytest.raises(FileExistsError):
            await view.chat([ChatMessage(role="user", content="task")])
        with pytest.raises(ValueError, match="Failed"):
            await view.chat([ChatMessage(role="user", content="task")])
        assert {p.name: p.read_bytes() for p in directory.iterdir()} == {"sentinel": b"retained"}
    finally:
        await provider.aclose()


@pytest.mark.parametrize("review", [False, True])
async def test_outer_guard_blocks_before_lazy_allocation(tmp_path, review):
    directory = tmp_path / "inputs"
    provider, _capture, _adapter = setup(directory, lambda _: pytest.fail("Unexpected dispatch"))
    messages = [ChatMessage(role="user", content="original")]
    expected = request_identity(messages, [], 2048).request_sha256
    guard = FirstRequestGuard(provider, expected, review_only=review)
    try:
        with pytest.raises(ValueError):
            await guard.for_conversation("session").chat(
                [ChatMessage(role="user", content="changed")], max_tokens=2048
            )
        assert not directory.exists() and guard.forwarded_requests == 0
    finally:
        await guard.aclose()


async def test_parent_link_drift_before_first_chat_rejects_without_external_write(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    directory = parent / "inputs"
    provider, _capture, _adapter = setup(directory, lambda _: pytest.fail("Unexpected dispatch"))
    external = tmp_path / "external"
    external.mkdir()
    parent.rmdir()
    parent.symlink_to(external, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="canonical"):
            await provider.for_conversation("session").chat(
                [ChatMessage(role="user", content="task")]
            )
        assert list(external.iterdir()) == []
    finally:
        await provider.aclose()


async def test_missing_normalized_pair_cannot_allocate_or_forward_wire(tmp_path):
    directory = tmp_path / "inputs"
    capture = RequestCapture(directory, lazy=True)
    transport = CapturedTransport(
        httpx.MockTransport(lambda _: pytest.fail("Unexpected dispatch")), capture
    )
    try:
        with pytest.raises(ValueError, match="pairing"):
            await transport.handle_async_request(
                httpx.Request("POST", "http://fixture/api/chat", json={})
            )
        assert not directory.exists()
        with pytest.raises(ValueError, match="Failed"):
            capture.begin([], None, 0.0, 2048)
    finally:
        await transport.aclose()
