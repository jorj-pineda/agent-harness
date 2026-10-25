# Invalid-hash feedback comparison results

**Control: 0/2 accepted artifacts and 0/2 completed-and-accepted. Treatment: 0/2
and 0/2.** All four cells failed on the second provider request, before malformed-
hash feedback or editing. The diagnostic did not activate, so this comparison does
not measure whether it helps recover from a malformed hash. Backend cause remains
unknown. Keep the original no-go, 8/10 target, defaults and open G2–G5.

All [protocol](HASH_FEEDBACK_PROTOCOL.md) cells ran once after PR #208's reviewed
[execution freeze](HASH_FEEDBACK_EXECUTION_FREEZE.md) merged. Control used clean
`e92f6a11043e6d7025de13e75fd0e01818c4f696`, treatment clean
`e416ce5b4afd9fc643d3a9613cb1618cb1d2a94a`. Reporting commits were not execution
runtimes. Both arms used actual CLI, one literal disposable workspace, fresh
independent stores, the original T02 request/criteria and unchanged limits.
[Structured results](../../results/2026-10-25-duodoro-hash-feedback-comparison.json)
retain every row, input/usage/progress observation, grade and cleanup record.

## Fixed-denominator outcomes

| Cell | Arm | Saved run state | Final completion/check fields | Independent artifact acceptance | Failed HTTP call duration |
|---|---|---|---|---|---:|
| C1 | Prior diagnostic | failed | absent | rejected | 180.032s |
| T1 | Complete-hash diagnostic | failed | absent | rejected | 180.023s |
| T2 | Complete-hash diagnostic | failed | absent | rejected | 180.027s |
| C2 | Prior diagnostic | failed | absent | rejected | 180.046s |

Each cell returned one response requesting read_file(AGENTS.md). The read completed;
then request two raised read_timeout with no returned normalized response. No exact-
edit or command-start event appears in the complete, untruncated tool journal.
Full artifacts remain unchanged, and external acceptance/regrades reject the original
unfixed timer behavior. There was no hash validation, corrected proposal, fallback
write, check or treatment feedback exposure to observe.

Failed runs have no final TurnResponse: completion/verification fields remain null,
rather than being assigned completed, incomplete or not_run. Run failure and the
absence of check execution are separate observations. Final tool-call records are
unavailable; the report labels that limit and uses retained input prefixes and
untruncated durable tool-progress metadata. Each session archive remains its initial
zero-turn/zero-response record, while the failed run retains its independent profile,
budget and provider-failure evidence. No final transcript or response was invented.

All four rows remain in their planned /2 arm denominators. No warm-up, new hint,
argument/input correction, follow-up message, replacement, extra coding turn,
manual patch repair or execution-setup correction occurred. A timeout did not become
a retry slot or permit a larger request/turn allowance.

## Input parity, activation and limits

Actual initial normalized/wire/profile identities matched the reviewed freeze.
Both recorded wire inputs match byte-for-byte for C1/T1 and C2/T2, through the
failed second request. Exact normalized records differ after request one; normalized
and wire equality are reported separately rather than rewriting call IDs or input
history to claim complete equivalence. The raw normalized/wire inputs remain local.

There was **no invalid-hash feedback activation** in either treatment cell. The
intervention changes a diagnostic that none of these runs reached. Equal rejection
counts therefore establish neither a feedback remedy nor a feedback regression.
They also cannot establish backend queueing, generation, serialization or another
root cause for the read timeouts. The earlier successful parser refactors and prior
T02 validation failures remain distinct, unchanged observations.

Eight requests were recorded, four returned responses, and four requests failed.
All aggregate usage totals remain unknown. Each known successful-response prefix
is 1,962 prompt/22 output tokens; those figures are not task totals or billing.
The roughly 180-second HTTP failure duration is a provider-boundary observation,
not backend generation time or end-to-end task latency. Original 300-second dispatch
limits remain separate from setup, review, storage, transport and cleanup timing.

One selected public task with two temperature-zero repeats per arm is not a general,
causal, held-out or two-model quality comparison. No new useful unattended envelope
or default change follows. The accurate diagnostic remains basic usability backed
by its existing offline tests; this cohort supplies no live recovery benefit claim.

## Outcome audit and preservation

Saved failed-run JSON matches SQLite and native CLI GET after an owned API restart.
Each cell has one run/session, one coding POST and no fork/reversal. Initial session
archives and absent final responses remain explicit. All eight private normalized/
wire pairs match saved request identities/allowances and production serialization
reconstructed offline with MockTransport. The audit verifies complete response/
failure journals, role/tool continuity and recorder bounds/0700–0600 permissions.
Input presence is not backend receipt or generation progress.

Original full archived trees, bytes/modes, bounded diffs/coverage, unchanged protected
files and grade/regrade outcomes match. A further independent regrade under the
pinned managed Node binary reproduced every rejection without artifact changes.
The final audit verified **104 outcome-evidence hashes**, both frozen runtime trees/
source hashes, all 20 frozen recipes and 1,799 preparation-evidence hashes, current
last workspace equality and Duodoro head/status/index/refs fingerprints. These are
scoped preservation checks, not a full-byte audit of Jorge's checkout or host isolation.

All **eight owned execution/review APIs exited**, port18761 is released and Gemma
is unloaded. The existing Ollama service remains. No browser, Duodoro working-checkout
edit/commit/push, additional or paid model, dependency installation or Docker action
occurred. Retained plaintext evidence, immutable execution recipes and the separate
reporting-only post-execution audit recipe are at:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-hash-feedback-live-6grynhsy`

The reporting recipe was developed after dispatch; its hash and additional audit
receipts are retained in the structured results. It is not a changed frozen input,
new inference or patch repair. It succeeded on its first execution. The earlier
copied staging-inode preparation correction stays labeled separately in the freeze.

## Disposition

This finite comparison is closed. It completed all four scheduled attempts and
preserved the evidence, but failed before its intervention could activate. Do not
reopen cells, silently retry, expand timeouts/budgets, promote a backend hypothesis
or call the diagnostic a measured task-quality improvement.

Original ten-task counts, the 8/10 target, release no-go/feature freeze and G2–G5
remain unchanged. This completes the requested protocol/freeze/results batch, not
the release roadmap. Further work needs a concrete retained-failure hypothesis or
an explicit narrower-envelope proposal with independent acceptance.

This docs/data report uses the frozen exact-runtime G0: control119 focused/1,453
offline, treatment 131/1,465, five live deselected each, Ruff/format/mypy and both
scripted 90/90 matrices per arm with labels/metric limits inspected. The new
settled-outcome audit is separate. The full suite was not rerun locally for this
report, and mechanics checks are not model-quality results.
