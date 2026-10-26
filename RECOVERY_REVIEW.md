# Bounded saved-run recovery review

`harness.recovery_review.build_recovery_review` deterministically extracts saved-run
evidence for review. The Beam read-window cohort left acceptable partial patches
without completed workflows; reviewers need the objective, observed diffs, failures
and recent actions before deciding how to recover. This extractor is the first
piece of that workflow. Owner-checked API access is now available; CLI access
follows separately.

The v1 packet retains the submitted objective, run/completion/check/workspace
statuses, saved stop/error text, up to eight failed/unavailable check observations,
the last eight tool actions, and the first eight saved diffs. Check flags retain
relevance and supersession; later passes do not silently erase earlier failures.
A returned tool is labeled `returned`, without inferring success. Missing final
responses leave action/diff/check counts and completion/check/workspace statuses
unknown; empty selected lists do not prove no work occurred. Running records are
not finalized evidence; callers must distinguish them before exposing stable review.

Each text excerpt records original UTF-8 byte length, SHA-256 and truncation. Text
is selected in objective, saved error/stop, check, recent-action and diff order, with
a shared 16,384-byte JSON-escaped text allowance and per-field limits (objective
4,096 bytes; ordinary text/diff 2,048; command/check error/workspace reason 1,024;
action name 128; action error/path/diff reason 512). Selection can leave an empty
excerpt with nonzero original length. Indices point to the original zero-based
response lists; omitted item counts remain explicit. Hashes identify source text,
not correctness. Inspect the original saved record when anything is omitted.

The complete compact Pydantic JSON encoding is at most 32,768 UTF-8 bytes, including
metadata. Oversized legacy/custom metadata raises an explicit error rather than
returning an unbounded packet. The packet pins the complete validated saved-record
serialization hash. This identifies that representation, not a database byte hash,
a workspace fingerprint or an independently verified claim. Presentation whitespace
may enlarge separately rendered JSON. Original records are unchanged.

This is saved, potentially untrusted repository/model evidence. Check commands,
errors and diffs can contain sensitive source text already present in the owner's
saved record. It is not a redacted public export. Raw tool arguments/results,
transcript and file spans are excluded. The packet does not read the current
workspace, create a plan, compact provider history, call a model, raise/reset a
budget or grant continuation/edit permission. Existing workspace/profile/transcript/
context checks remain necessary for any explicit continuation. Partial evidence
cannot replace authoritative raw history, current workspace review or task checks.

No measured completion gain, model-quality claim or release-readiness change follows
from deterministic extraction. Automatic compaction, model-generated summaries,
fresh file-span selection and execution using a packet remain separate work.

Validation for this extraction slice: five focused regressions and 1,568 offline
tests (five live deselected), Ruff/format/mypy pass. Offline extraction from all
eight retained Beam read-window run records yields 4,110–5,600-byte packets with
unchanged source bytes and original budget-exhausted states. This is deterministic
review compatibility, without replay, inference or evidence of recovery success.

## Read-only API access

`GET /runs/{run_id}/recovery-review?user_id=dev1` extracts the packet from the
committed saved run. It applies the same user-ID ownership check as run lookup
(the existing user-ID convention is not a new authentication mechanism). Missing
runs return 404, other owners 403, and missing/empty user IDs 422. Active runs
return 409 before extraction; oversized metadata returns 413 with no partial packet.
Finished, failed, interrupted and cancelled terminal records remain reviewable,
including those without a final response. No new packet table or schema migration
is needed: extraction is computed from authoritative saved evidence on each GET.

The endpoint does not resolve a workspace, construct tools, select a provider,
create a run or alter an archive. An API lifecycle restart returns the same packet
for unchanged saved records, even if the original workspace was moved or changed.
Stable equality applies to the same validated record and extraction version, not
unrelated future serializer/contract versions. The response includes the source
record hash; it is not a lease for editing later user work. Existing guarded
continuation requirements and the original saved history remain unchanged.

API validation: 51 focused regressions and 1,575 offline tests (five live excluded),
Ruff/format/mypy pass. Seven endpoint cases cover scripted HTTP/SSE partial-work
restart preservation, unknown failed/interrupted final evidence, active/owner/input
guards, control-character/Unicode wire bounds and explicit metadata refusal.
These are deterministic application-lifecycle checks, not live-model recovery or
an OS-process restart observation.
