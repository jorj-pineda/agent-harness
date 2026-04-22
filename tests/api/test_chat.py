"""End-to-end tests for the FastAPI server.

The acceptance test for step 10 is `test_facts_persist_across_sessions_for_same_user`:
turn 1 (session A) writes a fact via `remember_fact`; turn 2 (session B,
same user_id) sees that fact already injected into the system prompt
without the model having to call `recall_facts` first. That's the whole
point of `_refresh_facts_system_message` running per turn.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from api.server import build_components
from api.settings import Settings
from memory.store import FACTS_HEADING
from providers.base import ToolCall

from .conftest import Harness, make_response


def test_create_session_returns_id(harness: Harness) -> None:
    resp = harness.client.post("/sessions", json={"user_id": "u-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["session_id"], str) and body["session_id"]


def test_create_session_rejects_empty_user_id(harness: Harness) -> None:
    resp = harness.client.post("/sessions", json={"user_id": ""})
    assert resp.status_code == 422


def test_chat_returns_full_envelope(harness: Harness, make_session: Callable[[str], str]) -> None:
    session_id = make_session("u-1")
    harness.provider.script(make_response(content="hello there"))

    resp = harness.client.post(
        "/chat",
        json={"user_id": "u-1", "session_id": session_id, "message": "hi"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Rule-#5 envelope shape — every field present with sensible defaults.
    assert body["answer"] == "hello there"
    assert body["provider"] == "scripted"
    assert body["latency_ms"] > 0
    assert body["tool_calls"] == []
    assert body["memory_writes"] == []
    assert body["citations"] == []
    assert body["escalated"] is False
    assert body["confidence"] is None  # no search_docs call → ungrounded


def test_chat_404_when_session_unknown(harness: Harness) -> None:
    harness.provider.script(make_response(content="never reached"))
    resp = harness.client.post(
        "/chat",
        json={"user_id": "u-1", "session_id": "missing", "message": "hi"},
    )
    assert resp.status_code == 404
    # ScriptedProvider was never invoked; queue still primed.
    assert harness.provider.calls == []


def test_chat_403_when_user_id_does_not_match_session(
    harness: Harness, make_session: Callable[[str], str]
) -> None:
    session_id = make_session("owner")
    harness.provider.script(make_response(content="never reached"))
    resp = harness.client.post(
        "/chat",
        json={"user_id": "intruder", "session_id": session_id, "message": "hi"},
    )
    assert resp.status_code == 403
    assert harness.provider.calls == []


def test_chat_400_when_provider_unknown(
    harness: Harness, make_session: Callable[[str], str]
) -> None:
    session_id = make_session("u-1")
    harness.provider.script(make_response(content="never reached"))
    resp = harness.client.post(
        "/chat",
        json={
            "user_id": "u-1",
            "session_id": session_id,
            "message": "hi",
            "provider": "no-such-provider",
        },
    )
    assert resp.status_code == 400


def test_remember_fact_call_surfaces_in_memory_writes(
    harness: Harness, make_session: Callable[[str], str]
) -> None:
    session_id = make_session("u-mem")
    harness.provider.script(
        make_response(
            tool_calls=[
                ToolCall(id="t1", name="remember_fact", arguments={"fact": "prefers vegan"}),
            ],
            finish_reason="tool_use",
        ),
        make_response(content="got it"),
    )

    resp = harness.client.post(
        "/chat",
        json={"user_id": "u-mem", "session_id": session_id, "message": "I'm vegan"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer"] == "got it"
    assert body["memory_writes"] == ["prefers vegan"]


def test_facts_persist_across_sessions_for_same_user(
    harness: Harness, make_session: Callable[[str], str]
) -> None:
    """Acceptance test for step 10: same user_id, two session_ids, fact carries.

    Session A writes a fact; Session B (a fresh session, same user) should
    see that fact pre-injected into the system message at index 0 — without
    the agent having to call `recall_facts`.
    """
    user_id = "u-cross"

    # --- Session A: agent calls remember_fact, then answers.
    session_a = make_session(user_id)
    harness.provider.script(
        make_response(
            tool_calls=[
                ToolCall(id="t1", name="remember_fact", arguments={"fact": "size 9 shoe"}),
            ],
            finish_reason="tool_use",
        ),
        make_response(content="noted"),
    )
    resp_a = harness.client.post(
        "/chat",
        json={"user_id": user_id, "session_id": session_a, "message": "I wear a 9"},
    )
    assert resp_a.status_code == 200, resp_a.text
    assert resp_a.json()["memory_writes"] == ["size 9 shoe"]

    # --- Session B: brand-new session, same user. The system prompt should
    # already carry the fact even though session B has no prior turns.
    session_b = make_session(user_id)
    assert session_a != session_b
    harness.provider.script(make_response(content="welcome back"))

    pre_call_count = len(harness.provider.calls)
    resp_b = harness.client.post(
        "/chat",
        json={"user_id": user_id, "session_id": session_b, "message": "hi again"},
    )
    assert resp_b.status_code == 200, resp_b.text

    # Inspect what the harness handed the provider on session B's first call.
    new_calls = harness.provider.calls[pre_call_count:]
    assert len(new_calls) == 1
    messages_seen, _tools = new_calls[0]
    assert messages_seen[0].role == "system"
    assert FACTS_HEADING in messages_seen[0].content
    assert "size 9 shoe" in messages_seen[0].content


def test_facts_do_not_leak_between_users(
    harness: Harness, make_session: Callable[[str], str]
) -> None:
    """Cross-user isolation — alice's fact must not appear in bob's prompt."""
    # Alice remembers something.
    alice_session = make_session("alice")
    harness.provider.script(
        make_response(
            tool_calls=[
                ToolCall(id="t1", name="remember_fact", arguments={"fact": "alice secret"}),
            ],
            finish_reason="tool_use",
        ),
        make_response(content="ok"),
    )
    harness.client.post(
        "/chat",
        json={"user_id": "alice", "session_id": alice_session, "message": "remember this"},
    )

    # Bob chats — his system prompt must not mention alice's fact.
    bob_session = make_session("bob")
    harness.provider.script(make_response(content="hi bob"))
    pre = len(harness.provider.calls)
    harness.client.post(
        "/chat",
        json={"user_id": "bob", "session_id": bob_session, "message": "hi"},
    )
    messages_seen, _ = harness.provider.calls[pre]
    # Bob has no facts → no facts heading at all (system message is base prompt only).
    assert FACTS_HEADING not in messages_seen[0].content
    assert "alice secret" not in messages_seen[0].content


def test_system_message_is_replaced_not_duplicated_across_turns(
    harness: Harness, make_session: Callable[[str], str]
) -> None:
    """Two turns in the same session → still exactly one system message at index 0."""
    session_id = make_session("u-1")
    harness.provider.script(
        make_response(content="first"),
        make_response(content="second"),
    )

    harness.client.post(
        "/chat",
        json={"user_id": "u-1", "session_id": session_id, "message": "one"},
    )
    harness.client.post(
        "/chat",
        json={"user_id": "u-1", "session_id": session_id, "message": "two"},
    )

    # The provider's view on the second call: exactly one system message at the head.
    second_call_messages, _ = harness.provider.calls[1]
    roles = [m.role for m in second_call_messages]
    assert roles.count("system") == 1
    assert roles[0] == "system"


def test_build_components_smoke(tmp_path: Path) -> None:
    """build_components wires real backends from validated settings without raising.

    Doesn't touch the network — Ollama provider construction is lazy and
    Chroma is a local persistent client. Guards against signature drift in
    the default factory.
    """
    settings = Settings(
        default_provider="ollama",
        sqlite_db_path=tmp_path / "support.db",
        chroma_path=tmp_path / "chroma",
        memory_db_path=tmp_path / "memory.db",
    )
    components = build_components(settings)
    try:
        assert "ollama" in components.providers
        assert components.router.resolve(None).name == "ollama"
        assert components.fact_store is not None
    finally:
        components.fact_store.close()
