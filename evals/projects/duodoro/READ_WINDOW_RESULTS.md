# Beam default-read-window results

**Fixed cohort complete, 2026-10-26 UTC.** Eight actual CLI/shared-runtime attempts
compare default reads (C, 0) with an opt-in 80-line default window (W). Both arms
have **0/4 completed-and-independently-accepted workflows**. Independent artifact
acceptance is C **1/4**, W **3/4**. Every attempt stopped at a context or total-token
gate. This does not demonstrate improved completed-task quality or justify changing
the default. [Sanitized per-request rows](../../results/2026-10-26-beam-read-window-t02-t08-cli.json)
retain all attempts; earlier cohorts remain unchanged and are not reused as controls.

## Fixed attempts and independent acceptance

| Order / cell | Stop | Observed configured check | Artifact accepted | Completed and accepted | Requests | Reported total tokens |
|---|---|---|---|---|---:|---:|
| 1 / T02-C1 | Context | Not run | Yes | No | 5 | 41,346 |
| 2 / T02-W1 | Total tokens | Passed | Yes | No | 8 | 60,398 |
| 3 / T02-W2 | Total tokens | Not run | Yes | No | 7 | 51,170 |
| 4 / T02-C2 | Context | Failed | No | No | 5 | 42,311 |
| 5 / T08-W1 | Total tokens | Not run | Yes | No | 9 | 53,230 |
| 6 / T08-C1 | Context | Not run | No | No | 7 | 48,354 |
| 7 / T08-C2 | Context | Not run | No | No | 6 | 39,589 |
| 8 / T08-W2 | Total tokens | Not run | No | No | 10 | 54,319 |

Primary counts per task are **0/2 for each task/arm**. Artifact counts are T02:
C 1/2, W 2/2; T08: C 0/2, W 1/2. No failed or unfinished attempt is excluded.
A passing external patch check does not complete a workflow or imply answer-claim
correctness; observed configured checks and independent acceptance are distinct.

The accepted timer patches floor durations and pass unchanged external acceptance
and regrade. T02-C2 leaves the source unchanged. Its attempted direct `node` command
is refused by the existing allowlist; the observed configured check fails. T02-W1
has a passed check plus a later command that is not executed after the turn stops.
T08-W1 extracts and exports one shared `boundedDuration`, uses it for both fields,
and removes duplicate inline expressions while preserving fractions, Symbol errors,
defaults, bounds, mode/session validation and other parsers. It passes executable
acceptance/regrade and the original independent source rubric. Both control refactors
extract/use the helper but omit its required export; checks and source review reject
them. T08-W2 leaves the original file unchanged and fails both. Its `git_status`
reports that the disposable workspace has no `.git`; no Git repository is added.
These observations describe the final artifacts, without assigning a sole causal
explanation to a model or tool-interface difference. No patch repair is performed.

## Payload and instruction observations

Final **actual dispatched** input payloads (not estimates of the prevented next call):

| Cell | All message payload bytes | Tool-role payload bytes | Continuation bytes, subset |
|---|---:|---:|---:|
| T02-C1 | 40,114 | 31,355 | 6,215 |
| T02-W1 | 38,041 | 27,064 | 7,678 |
| T02-W2 | 34,119 | 31,410 | 596 |
| T02-C2 | 38,523 | 29,951 | 6,668 |
| T08-W1 | 30,715 | 22,908 | 4,172 |
| T08-C1 | 37,297 | 32,712 | 2,269 |
| T08-C2 | 36,766 | 33,991 | 471 |
| T08-W2 | 25,687 | 21,833 | 1,385 |

These use the unchanged `message_size_bytes` heuristic. Tool names/arguments and
serialized continuation contribute to the applicable role; continuation is already
included, not an additional total. Schemas/per-message token overhead are excluded
from this byte table; the raw report records schema estimates, request allowance,
saved prompt estimate and reported usage separately for every request. Bytes are
not provider tokenization, caching, billing or causal proof. Later model actions and
histories differ. W makes 34 requests and reports 219,117 total tokens; C makes 23
and reports 171,600. Smaller final inputs do not establish lower total resource use.

Successful observed `read_file` ranges for immutable original AGENTS.md cover all
265 lines in every control and T02-W2. T02-W1, T08-W1 and T08-W2 show only lines
1–80, with 81–265 unobserved through that tool. Exact whole-file hash, selected
range proofs and unread spans are retained per row. This supplementary observation
is not all-method reading, comprehension, compliance or an acceptance condition;
it does not authorize ignoring unread instructions. T02-W2 demonstrates that an
explicit later range can recover full observed coverage despite a default window.

## Execution, audit and billing scope

The [protocol PR #225](https://github.com/jorj-pineda/agent-harness/pull/225) and
[actual freeze PR #226](https://github.com/jorj-pineda/agent-harness/pull/226) were
independently reviewed at their exact heads, passed both CI checks and merged before
inference. Approval pins freeze head `6fd4e6fb7c8ffe3cae66f3e663eb8f401aaa606d`
and merge `e702382c4e9b83be15f4823c5b665690b4dd8bec`. Runtime is clean
`edb12647e77b9d59003fe6465a00996f0ef41d7e`; Duodoro source remains
`5324cabba4ad3c8ad2e1cd1480218cf087d9b741`, with original public task messages,
criteria and protected file modes. The prepared/unrun protocol and freeze documents
retain their historical preparation state; this report records subsequent execution.

Both arms select `beam-free` through the existing Go route and the `beam` profile.
Only the read-file description differs across first tool schemas/native bodies;
initial per-task messages/allowance match and exact first identities/native bodies
repeat within each task/arm. No schema/path/history/reasoning substitutions are made.
Compact/full runtime, empty independent stores, bounded reasoning replay, HTTP 180,
CLI 600, 300 dispatch seconds, 64,000 total/8,192 output tokens, 24 tools, 12 iterations,
2,048/minimum-256 request allowance, 16,384 context, ratio 3 and temperature 0 remain
fixed. Final review/storage/cleanup can exceed dispatch time. Durable created/finished,
loop latency and workspace-review durations have different scopes, not a hard
end-to-end deadline. Hosted revision, tokenizer, routing, caching and defaults remain unknown.

Fresh metadata/listing/declared-free pricing gates precede every cell. All **57**
returned responses have known reported usage and report numeric cost 0, billing
source `go`, estimated state absent; all pass the billing refusal guard before tools
or another call. No hosted stop-all error occurs. This is backend metadata, not
invoice or remaining-subscription verification. Post-response refusal cannot prevent
an already-incurred unexpected charge. Additional paid allowance remains $0; no
fallback, upgrade, new route/model/credential or generation warmup is used.

Each original keyed CLI submission occurs once. All eight full artifact archives,
external grades/regrades, source verdicts, input pairs/budget identities and one-run
SQLite stores pass final audit, with zero reversal/replacement. Native source and
review-only API restarts return identical saved evidence without new inference or
input appends. All **24 owned API lifecycles** settle and release the listener.
Duodoro's user checkout fingerprint is unchanged. No Docker, local model/service
action, coding followup, manual patch repair, retry or replacement cell occurs.

Private raw evidence, frozen preparation proofs, recipe and 438-file result evidence
index are retained at:
`/Users/jorge/.local/share/agent-harness/trials/duodoro-read-window-live-yx2ps4j8`.
The sanitized JSON pins the recipe, final audit and evidence-index hashes. It publishes
counts/hashes/statuses and source-review reasoning, without raw input bodies, tool
arguments/results, headers, credentials or account response bodies. Original full
artifact bytes/modes/types are covered by independently rechecked tree manifests.

The exact pinned runtime's pre-inference G0 passed 103 focused/1,563 offline tests
(five live excluded), Ruff/format/mypy and both labeled 90/90 scripted matrices.
Those are mechanics checks, separate from this live evidence. This results-only PR
validates arithmetic, first-input repeat identity, all retained hashes/archives,
privacy and documentation links; it changes no runtime behavior or criteria.

## Decision and remaining work

Keep unrestricted default 0. The window arm has more acceptable partial artifacts
in this small cohort but **no gain on the primary completion measure**. Both repeat
counts, public development tasks and changing hosted state preclude a general or
statistical quality claim. The original ten-task 8/10 target, release no-go, held-out
comparisons and broad G2–G5 remain open. A next product decision should address
observed context/total-token stops with bounded retained-evidence recovery or narrow
the supported task envelope; it needs its own scoped design and validation. No new
comparison, broader trial, automatic compaction or budget increase is authorized by
this report.
