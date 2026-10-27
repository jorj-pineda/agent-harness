"""Bounded immutable approval and atomic projected fork intent, without execution."""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from harness.config import Settings
from harness.recovery_identity import RecoverySource, recovery_messages_sha256
from harness.recovery_seed import build_recovery_seed
from harness.runs import RequestConflictError, RunInput, RunRecord
from harness.runtime import build_registry
from harness.session_store import SessionStore
from harness.state import Session
from memory import FactStore
from providers.base import ChatMessage
from tests.test_recovery_admission import approve
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
    # Install the complete known-profile source fixture into its authoritative rows.
    with store._conn:
        store._update_run(record)
        store._conn.execute(
            "UPDATE sessions SET archive=? WHERE session_id=?",
            (archive.model_dump_json(), archive.session.session_id),
        )
    seed = build_recovery_seed(
        record, archive, data["approval"], data["draft"], fork_session_id="recovery-fork"
    )
    request = RunInput(
        user_id="dev",
        session_id=seed.session_id,
        request_id="recover-once",
        message=data["approval"].message,
        provider="ollama",
        continuation_source=seed.continuation_source,
        recovery_source=seed.recovery_source,
    )
    data.update(seed=seed, input=request)
    try:
        yield data
    finally:
        facts.close()
        data["store"].close()


def create(data, request=None, fork=None):
    return data["store"].create_run(
        request or data["input"],
        provider="ollama",
        configured_model=data["record"].configured_model,
        fork=fork or (data["archive"], data["seed"]),
    )


def test_seed_and_run_commit_together_and_preserve_exact_source_after_restart(setup):
    data = setup
    source = data["record"].model_dump_json(), data["archive"].model_dump_json()
    seed = data["seed"]
    provenance = seed.recovery_source
    assert isinstance(provenance, RecoverySource)
    assert provenance.budget_scope == "separate-configured-turn"
    assert provenance.approval_json == data["approval"].model_dump_json()
    assert (
        provenance.approval_sha256 == hashlib.sha256(provenance.approval_json.encode()).hexdigest()
    )
    assert seed.turns == data["archive"].session.turns
    assert seed.inspection_only == data["archive"].session.inspection_only
    assert seed.messages == data["draft"].messages[:-1]
    assert (
        recovery_messages_sha256(
            [*seed.messages, ChatMessage(role="user", content=data["input"].message)]
        )
        == provenance.messages_sha256
    )
    record, created = create(data)
    assert created and record.request.recovery_source == provenance
    stored = data["store"].get(seed.session_id)
    assert stored.session == seed and stored.responses == data["archive"].responses
    assert (
        data["store"].get_run(data["record"].run_id).model_dump_json(),
        data["store"].get(data["archive"].session.session_id).model_dump_json(),
    ) == source
    path = data["store"].path
    data["store"].close()
    data["store"] = SessionStore(path)
    assert data["store"].get_run(record.run_id) == record
    assert data["store"].get(seed.session_id) == stored
    assert (data["root"] / "a.txt").read_bytes() == b"saved\r\nsecond\nlast"


def test_matching_replay_does_not_reseed_even_after_source_changes(setup):
    data = setup
    record, _ = create(data)
    changed = data["archive"].model_copy(deep=True)
    changed.session.messages[1].content += "later owner state"
    with data["store"]._conn:
        data["store"]._conn.execute(
            "UPDATE sessions SET archive=? WHERE session_id=?",
            (changed.model_dump_json(), changed.session.session_id),
        )
    again, created = data["store"].create_run(
        data["input"], provider="ollama", configured_model=None
    )
    assert not created and again == record
    assert len(data["store"].list_runs("dev")) == 2
    with pytest.raises(RequestConflictError):
        create(data, data["input"].model_copy(update={"message": "different"}))
    assert data["store"].get(changed.session.session_id) == changed


@pytest.mark.parametrize(
    "change",
    ["messages", "proposal", "turns", "root", "inspection", "request_source", "missing_source"],
)
def test_bad_projected_seeds_never_write_a_run_or_fork(setup, change):
    data = setup
    seed = data["seed"].model_copy(deep=True)
    request = data["input"]
    if change == "messages":
        seed.messages[1].content += "changed"
    elif change == "proposal":
        request = request.model_copy(update={"message": "changed"})
    elif change == "turns":
        seed.turns[0].user_input += "changed"
    elif change == "root":
        seed.workspace_root = str(data["root"] / "other")
    elif change == "inspection":
        seed = seed.model_copy(update={"inspection_only": True})
    elif change == "request_source":
        request = request.model_copy(
            update={
                "recovery_source": request.recovery_source.model_copy(
                    update={"source_turn_id": "other"}
                )
            }
        )
    else:
        request = request.model_copy(update={"recovery_source": None})
    with pytest.raises(ValueError):
        create(data, request, (data["archive"], seed))
    assert data["store"].get(seed.session_id) is None
    assert len(data["store"].list_runs("dev")) == 1


@pytest.mark.parametrize("change", ["run", "archive"])
def test_source_changes_between_preparation_and_transaction_abort_every_insert(setup, change):
    data = setup
    with data["store"]._conn:
        if change == "run":
            changed = data["record"].model_copy(update={"error": "later evidence"})
            data["store"]._update_run(changed)
        else:
            changed = data["archive"].model_copy(deep=True)
            changed.session.messages[1].content += "later evidence"
            data["store"]._conn.execute(
                "UPDATE sessions SET archive=? WHERE session_id=?",
                (changed.model_dump_json(), changed.session.session_id),
            )
    with pytest.raises(ValueError, match="source changed"):
        create(data)
    assert not data["store"]._conn.in_transaction
    assert data["store"].get(data["seed"].session_id) is None
    assert len(data["store"].list_runs("dev")) == 1
    actual = (
        data["store"].get_run(data["record"].run_id)
        if change == "run"
        else data["store"].get(data["archive"].session.session_id)
    )
    assert actual == changed


def test_session_insert_failure_rolls_back_run_insert_and_preserves_source(setup):
    data = setup
    before = data["store"].get(data["archive"].session.session_id)
    with data["store"]._conn:
        data["store"]._conn.execute(
            "CREATE TRIGGER refuse_fork BEFORE INSERT ON sessions "
            "WHEN NEW.session_id='recovery-fork' BEGIN SELECT RAISE(ABORT, 'fixture refusal'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        create(data)
    assert data["store"].find_run("dev", "recover-once") is None
    assert data["store"].get("recovery-fork") is None
    assert data["store"].get(data["archive"].session.session_id) == before
    assert not data["store"]._conn.in_transaction


def test_recovery_requires_one_fresh_fork_and_cannot_be_extended_as_an_ordinary_run(setup):
    data = setup
    with pytest.raises(ValueError, match="fresh atomic fork"):
        data["store"].create_run(data["input"], provider="ollama", configured_model=None)
    create(data)
    ordinary = RunInput(
        user_id="dev",
        session_id=data["seed"].session_id,
        request_id="bypass",
        message="another turn",
    )
    with pytest.raises(ValueError, match="single seeded run"):
        data["store"].create_run(ordinary, provider="ollama", configured_model=None)
    assert data["store"].find_run("dev", "bypass") is None


@pytest.mark.parametrize(
    "change",
    ["hash", "pin", "allowance", "ack", "duplicate", "noncanonical", "oversize", "nonfinite"],
)
def test_provenance_refuses_inconsistent_or_unbounded_approval(setup, change):
    body = setup["seed"].recovery_source.model_dump()
    payload = json.loads(body["approval_json"])
    if change == "hash":
        body["approval_sha256"] = "0" * 64
    elif change == "pin":
        body["messages_sha256"] = "0" * 64
    elif change == "allowance":
        payload["expected_request_output_allowance"] = True
    elif change == "ack":
        payload["acknowledge_omitted_context"] = 1
    elif change == "duplicate":
        body["approval_json"] = body["approval_json"].replace(
            '"message":', '"message":"other","message":', 1
        )
    elif change == "noncanonical":
        body["approval_json"] = json.dumps(payload, indent=2)
    elif change == "oversize":
        payload["padding"] = "🙂" * 17000
    else:
        payload["padding"] = float("nan")
    if change in {"allowance", "ack", "oversize", "nonfinite"}:
        body["approval_json"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if change != "hash":
        body["approval_sha256"] = hashlib.sha256(body["approval_json"].encode()).hexdigest()
    with pytest.raises(ValidationError):
        RecoverySource.model_validate(body)


def test_provenance_cannot_be_reassigned_on_session_or_run_input(setup):
    for value in (setup["seed"], setup["input"]):
        with pytest.raises(ValidationError):
            value.recovery_source = None


def test_absent_recovery_preserves_legacy_json_and_normalized_hashes():
    session = Session(session_id="s", user_id="dev", created_at=datetime(2026, 10, 26, tzinfo=UTC))
    expected_session = (
        '{"session_id":"s","user_id":"dev","checkpoint_turns":false,"inspection_only":false,'
        '"continuation_source":null,"workspace_root":null,"messages":[],"turns":[],"created_at":"2026-10-26T00:00:00Z"}'
    )
    assert session.model_dump_json() == expected_session
    request = RunInput(user_id="dev", session_id="s", message="Fix", request_id="once")
    expected_input = (
        '{"user_id":"dev","session_id":"s","message":"Fix","provider":null,'
        '"request_id":"once","continuation_source":null}'
    )
    assert request.model_dump_json() == expected_input
    record = RunRecord(request=request, provider="ollama")
    encoded = record.model_dump_json()
    assert '"recovery_source"' not in encoded
    assert RunRecord.model_validate_json(encoded).model_dump_json() == encoded
    assert Session.model_validate_json(expected_session).model_dump_json() == expected_session


def test_source_revalidation_is_inside_an_immediate_sqlite_write_lock(setup, monkeypatch):
    data = setup
    original_get = data["store"].get
    observed = []

    def get(session_id):
        if session_id == data["archive"].session.session_id:
            assert data["store"]._conn.in_transaction
            other = sqlite3.connect(data["store"].path, timeout=0)
            try:
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    other.execute(
                        "UPDATE sessions SET archive=archive WHERE session_id=?", (session_id,)
                    )
                observed.append(True)
            finally:
                other.close()
        return original_get(session_id)

    monkeypatch.setattr(data["store"], "get", get)
    record, created = create(data)
    assert created and observed == [True]
    assert data["store"].get_run(record.run_id) == record


def test_matching_key_committed_during_seed_preparation_is_joined_under_lock(setup, monkeypatch):
    data = setup
    original = data["store"]._fork_payload
    committed = []

    def prepare(source, session):
        payload = original(source, session)
        other = SessionStore(data["store"].path)
        try:
            record, created = other.create_run(
                data["input"],
                provider="ollama",
                configured_model=data["record"].configured_model,
                fork=(source, session),
            )
            assert created
            committed.append(record)
        finally:
            other.close()
        return payload

    monkeypatch.setattr(data["store"], "_fork_payload", prepare)
    record, created = create(data)
    assert not created and record == committed[0]
    assert len(data["store"].list_runs("dev")) == 2
