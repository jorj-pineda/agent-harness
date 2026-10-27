# Source wait versus fixed dispatch: results

**2026-10-27 — No benefit observed; provider waits did not bind.** All four fresh
original T02 cells ran in the [protocol](SOURCE_WAIT_PROTOCOL.md)'s fixed
60/180/180/60 order at [frozen](SOURCE_WAIT_EXECUTION_FREEZE.md) runtime
`4d57cc05fdfea5b33027af9b9aa9a074195320e9`. Freeze PR #253 merged as
`dbcdf2f5ccf059d513ebc4d3a80f82154f5507ee` after exact-head Luna review and both
CI checks at `950e61e165bb8bd7dde34be6f085fc8f2ccf69ce`; parent receipts and fresh
launch checks preceded generation. The
[sanitized rows and evidence hashes](../../results/2026-10-27-gemma-source-wait-interaction.json)
retain every outcome.

| Provider inactivity wait | Attempted | Accepted artifacts | Completed and accepted | Valid second-call responses |
|---|---:|---:|---:|---:|
| 60 seconds | 2/2 | 0/2 | 0/2 | 2/2 |
| 180 seconds | 2/2 | 0/2 | 0/2 | 2/2 |

There were no provider read timeouts or native wall-budget stops. Every cell
returned four valid responses; each adapter response duration was below 60 seconds.
The cohort did not reproduce the earlier source's second-call timeout, so it does
not test a demonstrated longer-wait rescue. Equal unsuccessful outcomes here do
not establish general equivalence or prove that the timeout can never matter.
The earlier stopped source, three unrun sources and eight unrun recovery arms stay
unchanged. This comparison has no C/P followups or measured recovery result.

## Actual stop and resources

Every native response has `completion_status=budget_exhausted`, checks `not_run`,
and the reason: the total-token budget could not fit another model request after
the proposed tools. Each cell executed three `read_file` calls: AGENTS.md lines
1–265/265, payloadParsers.js 1–174/174, then payloadParsers.test.js 1–167/167.
This is observed read coverage, not proof of comprehension or compliance with every
instruction. A fourth response proposed `emit_plan`; the shared pre-tool gate
recorded it as not executed because the turn had stopped. No edit or configured
verification ran. The tool trace contains four records, including that prevented
call; it is not four successful tool executions.

Each cell reports 26,325 prompt and 156 output tokens (26,481 total), with four
model-request attempts and known usage. The preventive next-request estimate can
stop below the 32K reported-token cap; unused tokens do not imply another complete
request could fit. Source saved-run elapsed spans 120.344–122.626 seconds, including
setup/storage. These observed upper bounds and reported counts satisfy the
conservative known phase-budget checks; they are not hard end-to-end guarantees
or backend tokenizer/billing attestation. Physical reported totals are 105,300
prompt and 624 output tokens across the four independent turns. No shared-source
cost is duplicated in this source-only comparison.

All eligible artifact files/modes remain initial. Original independent acceptance
and regrade reject each cell with exit 1; a separate post-execution copy reconstructed
from verified saved blobs also rejects each with exit 1. No reference solution or
manual patch was applied to live work. No expected outcome was forced and no
cell was replaced, retried, extended, repaired or given an extra message.

## Preservation and limits

Actual first normalized and exact wire inputs match the frozen captures in all
cells; saved profiles match their selected settings. All sixteen requests retain
paired normalized/wire evidence matching saved budget identities. Native transport
receipts bind the four HTTP timeout phases to the selected 60/180. Later histories
and tool identifiers are not forced equal; first-input parity is not backend
state equality or a causal guarantee.

Each cell began from the protected original image at the same owned root/inode,
with empty independent stores and a fresh metadata-verified owned backend. All
four original API/backend lifecycles and four GET-only restart APIs exit zero;
process groups are absent and owned ports 18843/11447 are free. Restart uses exact
closed-store clones and preserves actual store bytes, saved run/archive equality
and zero new provider captures. User Ollama on 11434 was not stopped or unloaded.
Duodoro head/status/index/refs and protected source/criteria/reference evidence
remain unchanged. Final frozen-base, environment, original/owned model and recipe/
evidence checks pass. Hashes bind the post-execution audit script and raw outcomes.

Private evidence:
`/Users/jorge/.local/share/agent-harness/trials/source-wait-live-6xawr27c`, especially
`cells/`, `live-operator.log`, parent receipts and `results-audit/`. Public metadata
contains no source/message bodies, headers, credentials or `.env` contents.
Process groups and path checks remain limited ownership checks, not OS isolation;
ignored content, concurrent writers and non-atomic staging retain the freeze's limits.

The exact-runtime 196-focused/2,178-offline/static/two-matrix G0 remains preparation
mechanics evidence. This reporting-only change used independent saved-artifact
regrades, input/profile/hash/row/accounting and cleanup checks; it did not rerun G0
or count scripted tests as model quality. Two temperature-zero repeats of one
public task are low-variation observations, not independent held-out evidence.

Close this finite comparison without adopting a timeout default or automatically
running another cohort. The observed obstacle here is the next-request total-token
gate after inspection; this is not proof that a new prompt, read window, token
allowance or recovery policy would improve acceptance. Original cohorts, no-go,
8-of-10 target and broad G2–G5 remain unchanged. No release or coding-quality gain
is established.
