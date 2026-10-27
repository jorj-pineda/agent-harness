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
