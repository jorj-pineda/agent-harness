# Guarded recovery execution

The prospective draft remains review-only. The explicit API described below can
submit one separately budgeted recovery turn after review. Shared admission and
durable provenance are prerequisites, not reusable execution leases. There is no
automatic compaction, retry, CLI submission or panel submission.

## Shared admission foundation (first slice)

`RecoveryApproval` pins the saved run representation, archived session, rendered
draft messages, current tool schema identity and positive first-request output
allowance, with the original selected file spans and explicit proposed message.
Three required acknowledgments accept uncaptured workspace state, omitted history
(including earlier assistant agreements) and an explicitly new turn budget. Each
requires literal boolean true, not numeric/string coercion. Neither acknowledgment
certifies semantic completeness or makes unknown workspace state safe.

`check_recovery_admission` must be called under the caller's process-local session/
workspace admission, with a newly captured current workspace manifest. It checks:

- owner/run/archive pins before saved-source access;
- valid linked final checkpoint and recorded before-ID pairing even with zero
  selected spans;
- identical scoped root identity, eligible regular-file bytes/hashes and modes;
- absence of newer running/failed/interrupted related or same-root archived work,
  excluding only the current admitted run ID supplied by the caller;
- a freshly rebuilt draft with known matching configuration and bounded estimated
  fit, plus explicit backend context configuration for Ollama;
- identical reviewed message/schema identities and output allowance.

It returns the exact draft only when these conditions hold. That return remains
`execution_supported=false`: it is not a reusable permission or stored lease. A
future execution caller must preserve the reservation through its governed work
and recheck the actual initial request. The caller must obtain authoritative owned
records and a trusted fresh capture, not accept a client-supplied current manifest.
Source records, archives, checkpoints, facts and user files are not modified.

Scope remains limited: matching capture is non-atomic against external writers;
ignored/environment contents, directory topology, ownership/ACLs/xattrs, tool/runtime
implementations and unrecorded backend/dependency state are not certified. Newer-work
association uses the existing source-session and exact canonical archived-root
convention, not cross-process locking or complete unknown-work discovery. Caller
exclusion is for its own current run only. Unrelated processes and writers do not
participate in process-local admission. Path/command validation is not isolation.

`RecoveryInputGuard` wraps one provider for one run. Its first chat compares exact
normalized message serialization, ordered schema identity and strict integer output
allowance before forwarding. Pure conversation binding retains the underlying
provider's routing and shared guard state. A mismatch permanently refuses that
wrapper; a raised exception/cancellation also prevents reuse. It never retries,
repairs, rewrites arguments, changes token accounting or owns/closes the underlying
client. After the matching first request, ordinary sequential loop updates can be
forwarded; ordinary runtime budgets and tool policies remain responsible for them.
These are normalized-input checks, not raw wire/backend/generation attestation.

Hashes reuse the existing ordered ChatMessage compact JSON joined by LF and sorted
compact schema JSON identities. Factoring these helpers preserves existing draft
and configured-profile representations. A rejected first request cannot certify
usage or completion, and an unknown forwarded failure stays unknown. No partial
model result is manufactured. Untrusted source/data framing and full user-text
retention are not a guarantee of model compliance or preservation of the meaning
of omitted assistant answers.

Any future recovery turn uses a separately acknowledged configured allowance. It
must not reset/transfer an old turn's remaining allowance, rewrite an old cohort,
or be counted as an original within-budget unattended success. Persisted approval,
atomic fork seeding, fresh execution integration, restart/keyed replay, explicit
client submission and measured recovery remain separate work. Existing defaults,
release no-go/8-of-10 targets and broader G2–G5 remain unchanged. No inference or
Duodoro edits occur in the foundation slice.

## Durable provenance and projected fork intent

`RecoverySource` is a bounded immutable storage record for a source run/session/turn,
source record/archive hashes, first messages/schema/allowance pins, draft version,
canonical approval JSON and its byte hash. The JSON is capped at 65,536 UTF-8 bytes;
metadata pins and literal scope acknowledgments must agree. This module validates
storage consistency only. Execution must parse it again as `RecoveryApproval` and
perform fresh admission; persistence is not a lease or confirmation that work ran.
The intent's budget scope is `separate-configured-turn`, not old allowance transfer.
Approval text is plaintext owner data already present in submitted review/input.

Optional provenance on `Session` and `RunInput` is frozen and omitted entirely when
absent, preserving legacy normalized JSON and hashes. Existing records are not
backfilled or rewritten. No table migration or extra plaintext source snapshot is
needed. Older/mixed-version programs are not supported as concurrent writers;
normal process-local/SQLite/filesystem limitations remain.

`build_recovery_seed` verifies source/draft/approval identities and prepares a new
checkpoint-enabled session with the source's unchanged turn records, owner, root,
inspection policy and response history. Only the projected provider messages change:
the proposed last user message is withheld so runtime can append it exactly once.
The original source archive retains all raw history; provenance points back to it.
The projected archive is not evidence that omitted assistant/tool context is irrelevant
or that partial history is semantically complete. The factory starts no execution.

`SessionStore.create_run` binds identical request/session provenance and the exact
seed-plus-proposed-message hash before inserting a recovery run. Source turn records,
root and inspection policy cannot be silently changed. An immediate SQLite write
transaction rereads authoritative owned source run/archive hashes before atomically
inserting the new run and fork archive. Source changes refuse, and an insertion error
rolls back both rows. The metadata lock is not a workspace writer lock. Existing
checkpoint blobs, source rows, user files and Git state remain untouched.

Matching keyed replay returns the existing run before reseeding or refreshing source
evidence, including after source changes. The key is rechecked under the write lock
for concurrent commit races; changed input conflicts instead of duplicating intent.
A new recovery request requires its atomic fresh fork, and an already seeded recovery
session rejects new ordinary run creation. Persisted seeds remain read-only on API
restart; this slice exposes no execution endpoint or automatic restoration.

Focused storage tests cover immutable bounded provenance, exact legacy serialization,
source preservation/restart, projected-message and identity refusals, source advancement,
run/session rollback, an actual competing SQLite write blocked during source reread,
a concurrently committed matching key joined under the lock, and one-run seed limits.
Fresh execution integration, actual checkpoint/provider sequencing, cancellation,
usage/first-input observation, generic-chat bypass prevention and explicit clients
remain the next separate work. No inference, model-quality gain, Duodoro edit or
release-gate closure follows from storage mechanics.

Provenance validation: 114 focused and 2,040 offline tests (five live deselected),
Ruff/format/mypy and diff checks passed. These are storage/coordination mechanics,
not model-quality or live recovery evidence. No provider inference was run.


## Explicit guarded one-turn API

`POST /runs/{run_id}/recover` accepts the complete reviewed `RecoveryApproval`,
plus `user_id`, a required `request_id` and optional configured `provider` selector.
The approval contains the proposed `message`, original `expected_record_sha256`,
explicit `spans`, `expected_archive_sha256`, `expected_messages_sha256`,
`expected_tool_specs_sha256`, `expected_request_output_allowance` and all three
literal-true acknowledgments above. Use the owner-checked recovery-draft preview
to inspect those inputs first. Do not substitute a preview's `estimated_fit` or
`execution_supported=false` field for the explicit approval. Extra fields refuse.
Existing caller-supplied user ownership is not a new authentication system.

The route returns HTTP 202 with a durable `RunRecord` intent, not a completion
claim. Inspect `GET /runs/{run_id}?user_id=...` and the saved fork archive for the
actual result. Source and fork IDs differ. Owner history, tool/budget observations,
checkpoint evidence and existing explicit cancellation remain available. The API
adds no streaming recovery endpoint or client workflow in this slice.

Preparation claims process-local source session/workspace admission, rebuilds the
current shared draft, checks the reviewed profile/context/pins, and atomically saves
the projected fork and run intent. Execution then acquires ordinary runtime
admission. Before binding its initial checkpoint it captures current scoped files
and repeats source/archive/checkpoint/newer-work/configuration/context/message checks.
A mismatch produces a blocked run with no provider dispatch; pre-intent refusals
create no run. A later race in actual prompt refresh is stopped by the first-input
provider guard before forwarding. External writers can still race after capture;
these checks do not provide process isolation or a cross-process workspace lock.

The runtime appends the proposal exactly once, refreshes current root instructions,
notes and configured check settings through shared helpers, and appends the same
recovery framing used in the reviewed draft. The execution profile treats that
framing as a custom logical base (`prompt_version=null`); it does not claim an
ordinary versioned prompt match. Immutable provenance binds the exact first
normalized messages, schemas and allowance. Normal token/tool/iteration/dispatch
wall budgets, inspection policy and tool validation govern all subsequent actions.
Missing usage remains unknown and can stop tools; a guard refusal/provider failure
never fabricates a final result. This new configured turn neither transfers old
allowances nor counts as an original within-budget unattended success.

A recovery fork permits only its seeded run. Ordinary chat, asynchronous run creation,
SSE chat, full-history continuation and nested recovery cannot extend it. Terminal
success, failure, cancellation and rejection retire its live session. Restart loads
review evidence without restoring an editing session. Matching keyed replay returns
the original run before refreshing or reseeding, even if files/source/settings have
since changed. Changed input on that key conflicts. A different key is a new explicit
attempt, subject to fresh guards; newer failed/interrupted related work can block it.
No automatic retry follows storage or provider failure.

Cancellation after tools settle retains partial changes, unknown interrupted usage
and bounded review evidence. Final persistence failure can leave real edits and
checkpoint observations without a saved final response; inspect before starting new
work. Checkpoint/diff evidence is bounded review, not a durable workspace transaction
or safe revert. Cleanup, snapshots and review can exceed dispatch deadlines.

Deterministic tests use fake inference and real tools in disposable repositories:
HTTP/SSE source runs survive restart; the exact reviewed first input is forwarded;
real writes and a configured pytest check are recorded; later user work, staged Git
state, modes and original source archives survive. Negative coverage includes
strict acknowledgments/ownership/pins, busy admission, changes between seed and
execution (files/modes/instructions/facts/configuration), actual first-input drift,
inspection restrictions, missing usage, generic-chat bypass, cancellation after an
edit, provider/final-storage failure and read-only replay. These are mechanics tests,
not evidence of model recovery quality or independent live-task acceptance.

CLI/panel explicit submission, operator usability and a separately frozen live
recovery comparison remain unfinished. Original cohorts, defaults, release no-go,
8-of-10 and broad G2–G5 gates remain unchanged. No inference or Duodoro edits ran.

API integration validation: 155 focused and 2,063 offline tests (five live deselected),
Ruff/format/mypy, diff and local documentation link checks pass. Both offline
scripted matrices passed 90/90 escalation decisions at unique temporary paths;
contract labels and metric limits were inspected. No model inference ran.
