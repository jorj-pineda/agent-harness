# Prospective recovery inputs

`harness.recovery_draft.build_recovery_draft` prepares a detached, inspectable
candidate request from an explicit [recovery-context selection](RECOVERY_CONTEXT.md).
Context/total-token gates stopped the retained Beam cohort even with acceptable
partial artifacts. Before trying a smaller request, reviewers need to see exactly
which instructions and evidence remain and whether the candidate fits configured
limits. This helper neither sends a request nor replaces stored history.

The source must have a final response matching the latest aligned session archive:
owner/session/run identity, finished unique turn IDs, response/configured-model/profile
identity and ordered presence of each structured user input are checked. Invalid
normalized tool-call/result history refuses instead of being repaired. Existing
saved-run hash/checkpoint selection guards still apply. A missing final response,
an older selected run whose session has advanced, or incomplete archive alignment
is unsupported; the older evidence-only preview remains available separately.

Every raw saved user-role message is copied verbatim in order, including any runtime
recovery guidance originally saved with that role. At most 64 messages and 32,768
UTF-8 user-text bytes are supported, with whole-input refusal instead of truncation.
The objective is also present with its original review excerpt/hash/truncation
metadata; a truncated excerpt never replaces full retained user-role text.
Assistant messages, tool results/calls and opaque assistant continuation are omitted,
with full source/user/omitted-nonuser message counts. The old system message is
replaced, and is included in the omitted count. These counts are message counts,
not source-token savings or proof that omitted evidence is irrelevant.

The candidate order is one current system message, all retained user-role messages,
one labeled historical JSON recovery-evidence message, then an explicit proposed
user message. The proposal must be nonblank and at most 4,096 UTF-8 bytes. The
system message uses the shared selected coding prompt and runtime refresh with
current inspection policy, configured check argv, workspace path, bounded owner
notes and opt-in complete root AGENTS.md loading. Unavailable enabled instructions
refuse. Parent/nested discovery and conventions previously read only through tools
are outside that loader. A fixed, versioned draft framing paragraph labels the
recovery JSON as untrusted historical data and requires applicable current-code/
instruction inspection before editing. Prompt labels do not enforce model compliance
or protect against all instruction injection. Normal runtime prompts stay unchanged.

Current tool definitions use the shared registry and inspection filter without
execution. The existing TokenBudget math computes the candidate prompt heuristic
and requested output allowance; blocked and unbounded-context states are explicit.
The comparison field estimates refreshed full source history plus the same proposal
under the same current schemas/ratio, without the draft framing paragraph. Either
estimate may be larger; neither is a historical sent-request token count, a measured
provider/tokenizer result, a wire-size estimate or a demonstrated efficiency gain.
Reported prompt anchors from old turns are not reused for these prospective inputs.

Budget scope is explicitly **hypothetical-new-turn**: current configured limits
are used only to describe a possible new turn. No old allowance is reset, transferred
or claimed unspent, no work is allocated, and fit cannot authorize continuation.
Configuration comparison retains matching/changed/unknown/not-recorded evidence;
matching is not compatibility or permission. Source record/archive, ordered messages,
recovery evidence and current schema hashes identify their validated representations,
not authenticity, workspace state or correctness. Message hashing joins each compact
ChatMessage JSON with LF; it does not identify a provider wire request or output cap.

The complete compact draft JSON is limited to 131,072 UTF-8 bytes, including nested
escaping/metadata. Oversize fails without partial output or silent evidence removal.
Source records, archive, checkpoints, facts and files remain unchanged. The draft
contains sensitive owner source/notes/messages and is not a redacted public export.
Refreshing instructions/notes reads current configured resources; it does not compare
current workspace bytes/modes, grant a lease or establish safe resumed execution.
Omitting earlier assistant answers/agreements may make short user replies ambiguous;
exact text retention is not semantic completeness. A reviewer must assess that risk.

The v1 result remains `read_only=true`, `execution_supported=false`. API/CLI access,
workspace/profile checks inside any future execution admission, durable provenance,
remaining-budget accounting for actual in-turn compaction and measured recovery
remain separate work. No live inference, default/policy change, model-quality gain,
release approval or broad milestone closure is claimed.

Validation: 70 focused draft/configuration/recovery regressions and Ruff/format/mypy
pass. Cases cover both shared prompt/toolset profiles, current root conventions,
scoped notes/check argv, exact user text, immutable source state, source/structure
refusals before blob reads, exact user-retention bounds, encoded aggregate overflow,
profile drift, inspection policy, prospective output/total/effective context limits
and a draft larger than the source estimate. Initial fixture errors (field/argument
names, frozen policy setup, shared response aliasing, minimum configured window and
which aggregate bound fires first) and literal constant typing were corrected
before the passing focused/static gate. Full offline validation is required before
merge; these deterministic checks do not measure live model recovery.

## Owner-checked API draft review

`POST /runs/{run_id}/recovery-draft` accepts `user_id`, `expected_record_sha256`,
`spans` and the explicit proposed `message`, plus an optional currently configured
`provider`. It carries no request key or execution flag, creates no run/journal/
checkpoint, and returns no executable seed or continuation lease. Unknown fields
are rejected. The same source selection as recovery-context is used; requests for
missing final responses or older runs whose session archive has advanced refuse.

The run and archive ownership convention is checked before current registry/provider
review. Active runs and overlapping session/workspace work refuse with 409 before
resource construction. Shared process-local admission covers the whole current
registry, instruction/note refresh, saved-source read and estimate computation;
all error paths release it. There is no cross-process/external-writer lock. The
existing lifecycle-owned FactStore stays on its event-loop thread. Review is
synchronous under admission, like configuration review; no blocking I/O deadline,
OS isolation or atomic comparison of all current resources is implied.

Success returns `Cache-Control: no-store` because complete current instructions,
notes, user text and selected source are sensitive owner data. Repeating the review
after API lifecycle restart retains identical draft bytes when relevant current
resources/settings are unchanged. Changed root instructions produce a new system/
message hash and configuration drift while original saved evidence stays unchanged.
Later source-file edits do not invalidate historical selections; this endpoint does
not perform workspace comparison and still cannot grant execution permission.
Unlike the historical-only preview, constructing current file-tool definitions needs
a usable configured workspace. Missing current resources return generic 503;
existing historical `recovery-context` review remains usable without the workspace.

Missing/foreign owned records return 404/403; malformed selection/proposal fields
422. Active/busy/stale/mismatched archive/source or unavailable enabled instructions
return fixed 409 codes. User-history/source/span/complete-draft bounds return 413;
invalid selected source UTF-8 415; uncaptured paths 404/nonexistent ranges 422.
SQLite/resource errors are generic 503 and inconsistent saved evidence is generic
409, without private exception text or partial draft output. A 200 response may
report blocked or unbounded estimated context; none of these states enables execution.

Scripted HTTP/SSE partial-edit regressions cover exact restart draft bytes, refreshed
instructions/drift, later edit and archive/run/checkpoint preservation, no additional
model calls/runs, owner/active/busy/complete-admission guards, source archive advance,
invalid proposal fields, aggregate user-history/instruction limits, storage privacy/
reservation release and missing-current-workspace versus historical-only access.
Negative record run-ID, saved profile, configured-model and provider cases close the
additional identity-test coverage suggested by the core PR review. CLI access and
actual execution/recovery remain separate; original release gates stay unchanged.

## CLI draft review

The existing recovery-context selection file (exactly `expected_record_sha256`
and `spans`) can be reused with an explicit proposed message:

```sh
agent-harness recovery-draft <run-id> --user-id dev1 \
  --selection-file /path/to/selection.json \
  --message "Finish the original bounded task"
```

An optional `--provider` selects only a currently configured server provider for
review; it is not a switch on the saved run or an inference request. Add `--json`
for the complete response. The same bounded regular UTF-8 selection-file reader
and duplicate-key rules as recovery-context apply. The proposed message is nonblank
and at most 4,096 UTF-8 bytes. Client input errors reject before HTTP.

The thin command sends one 30-second POST to `recovery-draft`, with no metadata GET,
retry, fallback, continuation submission or local project-source read. Generic
error categories suppress private input/transport/server details and explicitly
say no continuation was submitted. A missing current workspace may still permit
separate historical-only review; this command does not automatically request it.

Before either output mode, the client validates supported version, run/selection/
proposal identity, required hashes, non-execution and hypothetical-budget scope,
message roles/shape/counts/omissions and message hash. It parses the labeled embedded
historical packet through the existing recovery/context/span validators without
extra requests, checks the embedded evidence identity and supported final response
status, and validates context/configuration/instruction metadata shapes and status
consistency. Estimates and allowances stay server computations; the client does not
recompute or apply budgets. Unsupported or contradictory responses refuse without
printing partial draft contents. Reused validation/input-reader extraction preserves
the existing recovery-review/context/span commands.

Human output leads with original completion/check state and separates current
prospective estimates, profile drift and instruction loading. It shows omitted
message counts, exact candidate messages/hashes and the hypothetical-new-turn scope.
Estimated fit is explicitly not permission to execute; unbounded/blocked states
remain visible. Both human and JSON output escape owner text and terminal controls;
separate display whitespace/escaping can enlarge the output beyond the compact
server-response bound. These consistency checks cannot authenticate storage, prove
that user history was complete, validate fresh workspace state, or independently
verify server estimates/schema bindings. The server remains authoritative.

Validation: 317 focused regressions and Ruff/format/mypy, installed command help
and diff checks pass. Ninety-four new CLI cases include identity/hash/role/tool-call/
omission/context/configuration/instruction contradictions, explicit blocked and
unbounded review, safe rendering, private errors, pre-HTTP refusal and one-review-
request behavior. A real CLI-to-restarted-API test retains exact draft/run/archive
evidence and later user edits with zero new runs/provider calls. Full offline
validation is required before merge. This completes only prospective construction/
API/CLI review, not automatic compaction, safe execution or measured recovery.
Defaults, original cohorts, release no-go/8-of-10 and broad G2–G5 remain unchanged;
no inference, Duodoro edit, paid call or model-quality claim follows.
