"""Fresh recovery checks never grant a reusable lease or reset an original allowance."""

import asyncio
import hashlib

import pytest
from pydantic import ValidationError

from harness.checkpoints import capture_checkpoint
from harness.config import Settings
from harness.execution_profile import capture_execution_profile, tool_specs_digest
from harness.recovery_admission import (
    RecoveryAdmissionBlocked,
    RecoveryApproval,
    RecoveryInputGuard,
    check_recovery_admission,
)
from harness.recovery_draft import RecoveryDraftSelection, build_recovery_draft
from harness.recovery_identity import recovery_messages_sha256
from harness.runs import RunInput
from harness.runtime import build_registry
from harness.state import Session
from memory import FactStore
from providers.base import ChatMessage, ProviderError, ToolSpec
from tests.api.conftest import make_response
from tests.test_recovery_draft import prepared


@pytest.fixture
def setup(tmp_path):
    root, store, record, archive, request = prepared(tmp_path)
    facts = FactStore(tmp_path / "facts.db")
    settings = Settings(
        _env_file=None,
        max_context_tokens=16384,
        ollama_num_ctx=16384,
        max_request_output_tokens=1024,
        max_total_tokens_per_turn=64000,
    )
    registry = build_registry(fact_store=facts, user_id="dev", workspace_root=str(root))
    data = {
        "root": root,
        "store": store,
        "record": record,
        "archive": archive,
        "request": request,
        "facts": facts,
        "settings": settings,
        "registry": registry,
    }
    approve(data)
    try:
        yield data
    finally:
        facts.close()
        store.close()


def approve(data):
    record, archive, settings, registry = (
        data[k] for k in ("record", "archive", "settings", "registry")
    )
    profile = capture_execution_profile(settings, "ollama", registry)
    record.provider = "ollama"
    record.configured_model = profile.configured_model
    record.execution_profile = profile
    record.response.provider = "ollama"
    saved = archive.responses[-1]
    saved.response = record.response.model_copy(deep=True)
    saved.execution_profile = profile.model_copy(deep=True)
    saved.configured_model = record.configured_model
    request = RecoveryDraftSelection(
        **data["request"].model_dump(exclude={"expected_record_sha256"}),
        expected_record_sha256=hashlib.sha256(record.model_dump_json().encode()).hexdigest(),
    )
    draft = build_recovery_draft(
        data["store"],
        record,
        archive,
        request,
        settings=settings,
        provider="ollama",
        registry=registry,
        fact_store=data["facts"],
        user_id="dev",
    )
    data["draft"] = draft
    data["approval"] = RecoveryApproval(
        **request.model_dump(),
        expected_messages_sha256=draft.messages_sha256,
        expected_archive_sha256=draft.source_archive_sha256,
        expected_tool_specs_sha256=draft.tool_specs_sha256,
        expected_request_output_allowance=draft.context.request_output_allowance or 1,
        acknowledge_uncaptured_state=True,
        acknowledge_omitted_context=True,
        acknowledge_new_turn_budget=True,
    )
    data["current"] = capture_checkpoint(data["root"], "prospective-fork").record


def check(data, **kwargs):
    return check_recovery_admission(
        data["store"],
        data["record"],
        data["archive"],
        data["current"],
        data["approval"],
        settings=data["settings"],
        provider="ollama",
        registry=data["registry"],
        fact_store=data["facts"],
        user_id="dev",
        **kwargs,
    )


def test_matching_fresh_workspace_profile_and_review_return_exact_draft_without_mutation(setup):
    data = setup
    before = data["record"].model_dump_json(), data["archive"].model_dump_json()
    result = check(data)
    assert result == data["draft"]
    assert result.execution_supported is False
    assert result.budget_scope == "hypothetical-new-turn"
    assert result.context.request_output_allowance == 1024
    assert (data["record"].model_dump_json(), data["archive"].model_dump_json()) == before
    assert (data["root"] / "a.txt").read_bytes() == b"saved\r\nsecond\nlast"


@pytest.mark.parametrize(
    "change,code",
    [
        ("owner", "source_unavailable"),
        ("active", "run_active"),
        ("record", "record_changed"),
        ("archive", "archive_changed"),
        ("missing", "checkpoint_unavailable"),
        ("content", "workspace_changed"),
        ("mode", "workspace_changed"),
        ("root", "workspace_changed"),
    ],
)
def test_source_and_workspace_rejections_precede_blob_access(setup, monkeypatch, change, code):
    data = setup
    if change == "owner":
        data["record"].request.user_id = "other"
    elif change == "active":
        data["record"].status = "running"
    elif change == "record":
        data["record"].error = "changed"
    elif change == "archive":
        data["archive"].session.messages[1].content = "changed"
    elif change == "missing":
        monkeypatch.setattr(data["store"], "get_checkpoint", lambda _: None)
    elif change == "content":
        data["current"].files["a.txt"].sha256 = "0" * 64
    elif change == "mode":
        data["current"].files["a.txt"].mode ^= 0o100
    else:
        data["current"].root_identity = (0, 0)
    monkeypatch.setattr(
        data["store"], "checkpoint_content", lambda *args: pytest.fail("No blob read")
    )
    with pytest.raises(RecoveryAdmissionBlocked, match=code):
        check(data)


@pytest.mark.parametrize(
    "field",
    ["expected_messages_sha256", "expected_tool_specs_sha256", "expected_request_output_allowance"],
)
def test_changed_review_pin_cannot_pass_fresh_check(setup, field):
    data = setup
    data["approval"] = data["approval"].model_copy(
        update={field: 1 if field.endswith("allowance") else "0" * 64}
    )
    with pytest.raises(RecoveryAdmissionBlocked, match="draft_changed"):
        check(data)


@pytest.mark.parametrize(
    "case,code",
    [
        ("drift", "configuration_unusable"),
        ("unknown", "configuration_unusable"),
        ("unbounded", "context_unusable"),
        ("short_context", "context_unusable"),
        ("unknown_backend", "backend_context_unknown"),
    ],
)
def test_unknown_drift_or_unusable_context_is_not_execution_permission(setup, case, code):
    data = setup
    if case == "drift":
        data["settings"] = data["settings"].model_copy(update={"max_tool_calls_per_turn": 1})
    elif case == "unknown":
        data["record"].execution_profile = None
        data["archive"].responses[-1].execution_profile = None
        data["approval"] = data["approval"].model_copy(
            update={
                "expected_record_sha256": hashlib.sha256(
                    data["record"].model_dump_json().encode()
                ).hexdigest(),
                "expected_archive_sha256": hashlib.sha256(
                    data["archive"].model_dump_json().encode()
                ).hexdigest(),
            }
        )
    else:
        data["settings"] = data["settings"].model_copy(
            update={
                "max_context_tokens": 256
                if case == "short_context"
                else 0
                if case == "unbounded"
                else 16384,
                "ollama_num_ctx": None,
            }
        )
        approve(data)
    with pytest.raises(RecoveryAdmissionBlocked, match=code):
        check(data)


def test_unknown_newer_work_blocks_and_only_the_named_current_run_can_be_excluded(setup):
    data = setup
    run, _ = data["store"].create_run(
        RunInput(
            user_id="dev",
            session_id=data["record"].request.session_id,
            request_id="newer",
            message="unsettled",
        ),
        provider="ollama",
        configured_model="gemma",
    )
    with pytest.raises(RecoveryAdmissionBlocked, match="newer_unfinished_work"):
        check(data)
    assert check(data, excluding_run_id=run.run_id) == data["draft"]
    data["store"].fail_run(run.run_id)
    with pytest.raises(RecoveryAdmissionBlocked, match="newer_unfinished_work"):
        check(data)


def test_same_root_unknown_work_from_another_owner_blocks(setup):
    data = setup
    other = Session(user_id="other", workspace_root=str(data["root"]))
    data["store"].save(other)
    data["store"].create_run(
        RunInput(
            user_id="other", session_id=other.session_id, request_id="foreign", message="work"
        ),
        provider="ollama",
        configured_model="gemma",
    )
    with pytest.raises(RecoveryAdmissionBlocked, match="newer_unfinished_work"):
        check(data)


@pytest.mark.parametrize(
    "field",
    ["acknowledge_uncaptured_state", "acknowledge_omitted_context", "acknowledge_new_turn_budget"],
)
@pytest.mark.parametrize("value", [None, False, 1, "true"])
def test_all_scope_acknowledgments_require_explicit_boolean_true(setup, field, value):
    body = setup["approval"].model_dump()
    body[field] = value
    with pytest.raises(ValidationError):
        RecoveryApproval.model_validate(body)


class RecordingProvider:
    name = "ollama"

    def __init__(self):
        self.calls = []
        self.bindings = []
        self.failure = None

    def for_conversation(self, conversation_id):
        self.bindings.append(conversation_id)
        return self

    async def chat(self, messages, *, tools=None, temperature=0.0, max_tokens=None):
        self.calls.append((messages, tools, temperature, max_tokens))
        if self.failure is not None:
            raise self.failure
        return make_response(content="Done")


@pytest.mark.asyncio
async def test_first_input_is_exact_then_normal_loop_updates_pass_without_rewriting(setup):
    data = setup
    provider = RecordingProvider()
    guard = RecoveryInputGuard(provider, data["approval"])
    bound = guard.for_conversation("owned-fork")
    messages = data["draft"].messages
    specs = data["registry"].as_tool_specs()
    assert tool_specs_digest(specs) == data["draft"].tool_specs_sha256
    assert recovery_messages_sha256(messages) == data["draft"].messages_sha256
    first = await bound.chat(messages, tools=specs, max_tokens=1024)
    assert first.content == "Done"
    later = [*messages, ChatMessage(role="assistant", content="next state")]
    await bound.chat(later, tools=specs, max_tokens=5)
    assert provider.bindings == ["owned-fork"]
    assert provider.calls == [(messages, specs, 0.0, 1024), (later, specs, 0.0, 5)]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["message", "schema", "allowance", "boolean", "missing"])
async def test_first_input_drift_prevents_forwarding_and_permanently_refuses_this_wrapper(
    setup, change
):
    data = setup
    provider = RecordingProvider()
    guard = RecoveryInputGuard(provider, data["approval"])
    messages = [m.model_copy(deep=True) for m in data["draft"].messages]
    specs = data["registry"].as_tool_specs()
    allowance = 1024
    if change == "message":
        messages[0].content += "changed"
    elif change == "schema":
        specs = [ToolSpec(name="other", description="different", parameters_schema={})]
    elif change == "allowance":
        allowance = 2
    elif change == "boolean":
        allowance = True
    else:
        allowance = None
    with pytest.raises(ProviderError, match="dispatch refused"):
        await guard.chat(messages, tools=specs, max_tokens=allowance)
    with pytest.raises(ProviderError, match="further dispatch"):
        await guard.for_conversation("other-view").chat(
            data["draft"].messages, tools=data["registry"].as_tool_specs(), max_tokens=1024
        )
    assert provider.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ProviderError("unknown outcome"), asyncio.CancelledError()])
async def test_failed_or_cancelled_forwarding_never_retries_or_reopens_the_guard(setup, failure):
    data = setup
    provider = RecordingProvider()
    provider.failure = failure
    guard = RecoveryInputGuard(provider, data["approval"])
    with pytest.raises(type(failure)):
        await guard.chat(
            data["draft"].messages, tools=data["registry"].as_tool_specs(), max_tokens=1024
        )
    provider.failure = None
    with pytest.raises(ProviderError, match="further dispatch"):
        await guard.chat(
            data["draft"].messages, tools=data["registry"].as_tool_specs(), max_tokens=1024
        )
    assert len(provider.calls) == 1
