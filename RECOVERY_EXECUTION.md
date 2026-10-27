# Guarded recovery execution: admission foundation

The prospective draft is review-only. Executing a smaller history requires fresh
workspace/configuration/context checks, durable provenance and exact first-input
binding. This first slice adds shared checks and a provider-neutral dispatch gate;
no API, CLI, runtime option or recovery execution path is enabled yet.

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
Duodoro edits occur in this slice.

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
