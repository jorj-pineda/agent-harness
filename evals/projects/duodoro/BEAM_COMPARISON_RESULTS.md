# Original-task Gemma / Beam comparison results

**All eight fixed cells completed once, 2026-10-26 UTC.** Local `gemma4:12b`
produced **2/4 completed-and-independently-accepted artifacts**; Go `beam-free`
produced **1/4**. These are observations on two public development tasks with two
repeats each, not a general model ranking, held-out gain or release approval.
The original no-go, 8/10 target, historical cohorts, defaults and G2–G5 stay unchanged.

[Sanitized per-cell report](../../results/2026-10-26-gemma-beam-original-t02-t08-cli.json)
retains every planned row, usage uncertainty, request counts, billing observations,
source verdicts and evidence hashes. The [protocol](BEAM_COMPARISON_PROTOCOL.md) and
[execution freeze](BEAM_COMPARISON_EXECUTION_FREEZE.md) retain their historical unrun
preparation status; this results record does not authorize additional attempts.

## Counts and outcomes

| Task/model | Completed and accepted | Artifact accepted | Planned/attempted |
|---|---:|---:|---:|
| T02 timer / Gemma | 0/2 | 0/2 | 2/2 |
| T02 timer / Beam | 1/2 | 1/2 | 2/2 |
| T08 refactor / Gemma | 2/2 | 2/2 | 2/2 |
| T08 refactor / Beam | 0/2 | 0/2 | 2/2 |

| Fixed order | Outcome | Observed check | Independent artifact |
|---|---|---|---|
| T02-G1 | Failed: second request read timeout | Absent final response | Rejected; no edit |
| T02-B1 | Incomplete: truncated response | Not run | Rejected; no edit |
| T02-B2 | Completed | Passed | Accepted |
| T02-G2 | Failed: second request read timeout | Absent final response | Rejected; no edit |
| T08-B1 | Budget exhausted: next request prevented | Not run | Rejected; source review failed |
| T08-G1 | Completed | Passed | Accepted; source review passed |
| T08-G2 | Completed | Passed | Accepted; source review passed |
| T08-B2 | Budget exhausted: further work prevented | Not run | Rejected; source review failed |

Each settled full artifact was copied and independently graded/regraded outside the
editable tree. Protected bytes/modes stayed unchanged. Each T08 verdict records the
original rubric and final diff hashes. Static artifact inspection by Codex accompanied
executable evidence; no answer-text similarity or model-only acceptance score was used.
The two successful refactors export one boundedDuration helper, route both fields
through it, remove the duplicated inline implementations and preserve fractions,
conversion errors, defaults/clamps, modes and other parser behavior. Both failed
Beam refactors leave the initial file unchanged, with no required helper/export/reuse.
The accepted timer patch floors the existing clamped values and changes only the
allowed parser file. Complete bounded diffs have no omitted paths or truncation.

## Failure observations and resources

Both Gemma timer attempts returned a read_file proposal, then their second HTTP
request reached the configured 180-second read timeout. Their initial artifacts,
failed durable runs, absent TurnResponse and known first-response usage survive
restart. Total tokens and final-response tool counts remain unknown, not zero.
Persisted tool-progress metadata separately records the observed read.

Beam T02-B1 returned four responses and stopped on a length-truncated response,
without edits. T08-B1 proposed one exact replacement whose old_text used literal
backslash-n characters where the file had newlines; validation refused the edit.
After further inspection, the total-token estimate gate prevented another request.
T08-B2 inspected files but reached the token gate before editing; a later proposed
git_status was marked unexecuted after the stop. This is a prevented tool, not a
failed executed Git command. Token gates reserve estimated next-request work;
“budget exhausted” does not mean reported usage reached exactly 64,000.

| Cell | Model requests | Reported total tokens | Output tokens | Saved run elapsed seconds |
|---|---:|---:|---:|---:|
| T02-G1 | 2 | Unknown | Unknown | 205.92 |
| T02-B1 | 4 | 29,108 | 2,197 | 10.26 |
| T02-B2 | 6 | 53,352 | 2,244 | 11.85 |
| T02-G2 | 2 | Unknown | Unknown | 187.43 |
| T08-B1 | 6 | 39,408 | 1,312 | 7.40 |
| T08-G1 | 8 | 46,419 | 1,125 | 154.86 |
| T08-G2 | 8 | 46,421 | 1,127 | 156.05 |
| T08-B2 | 7 | 52,725 | 457 | 7.06 |

Saved-run elapsed uses durable created/finished timestamps; it is not the full CLI,
API lifecycle, grading or cleanup duration. The fixed allowance remained 300 dispatch
seconds/cell, 64,000 reported total/8,192 output tokens, 24 tools and 12 iterations;
request cap 2,048/minimum 256, context 16,384, ratio 3 and temperature 0 stayed fixed.
Provider timeout 180/CLI inactivity 600 were experimental, not changed defaults.
No budget transfer, warm-up, hidden retry, new coding message, repair, replacement
or fallback ran. Configured completion feedback was part of the frozen shared loop.

All **23 returned Beam responses** reported finite cost zero, billing_source=go
and no estimated field, passing the evaluation-only guard. All four hosted aggregate
reported costs are zero; local costs are unpriced. This is backend-reported billing,
not invoice verification or a guarantee about future requests. Public metadata/user
attestation was rechecked as specified but is not independent entitlement/allowance
proof. The post-response guard cannot prevent an already-incurred unexpected charge.
No hosted access/rate/usage/billing error triggered the stop-all rule in this cohort.

All returned Beam responses included bounded continuation state. Typed normalized
inputs and actual wire captures pair with all 43 request attempts; offline native
adapter serialization reproduces their bodies, including subsequent replay. Initial
normalized parity holds per task and exact first bodies match within each provider's
repeats. Native provider bodies and later histories differ. These observations do
not identify the hosted served revision, default reasoning policy, tokenizer,
caching behavior or backend cause of any outcome.

## Execution integrity and cleanup

Runtime `de6083d1c69afb715d212704f3c25742b596e9f1` and Duodoro source
`5324cabba4ad3c8ad2e1cd1480218cf087d9b741` match the reviewed freeze. Luna found
no blockers at freeze head `7d28b70`; both CI checks passed before #220 merged.
Only then did the parent write the launch approval. One prelaunch operator check
failed to discover the GitHub keyring because HOME was stripped; supplying HOME to
the operator/gh process fixed discovery. No cell/store/input/submission existed at
that failure. API and CLI child environments remained the frozen stripped version;
no secret value was added. The incidental gh device metadata was retained outside
the checkout. This was an operator setup correction, not a repeated model attempt.

Each cell has exactly one original keyed run and no reversal. Before common-root
replacement, full archive/input/review equality and complete audits passed. Default
source and outer review-only API restarts returned identical saved run/session data
without new provider calls or modified input files. All **24 actual owned API
lifecycles** settled and released the listener. Final checks reverified all eight
archives, saved review/input hashes, one-run SQLite records and absent reversals.
Evaluation-loaded Gemma was unloaded; Jorge's Ollama service remains. No Docker,
dependency installation, Duodoro source edits/commits/pushes or manual patch rescue.

Retained evidence root:
`/Users/jorge/.local/share/agent-harness/trials/duodoro-beam-live-1bt9rl3i`.

The report pins the final local evidence index and packaging recipe. Raw requests,
tool arguments/results, full trees, stores and server logs remain local; the public
report uses allowlisted observations/progress and hashes. This is host-specific
retention, not a portable public evidence bundle. Packaging drafts were retained
privately; no inference or artifact was replayed to prepare the report.

The three-PR protocol/freeze/results slice is complete. This sample shows a timer
success on Beam and two refactor successes on Gemma under the chosen limits, with
five unsuccessful cells retained. It does not satisfy the ten-task personal-use
release target, held-out repeated comparisons or broad G2–G5. Further work should be
chosen from these concrete failures rather than declaring the harness ready or
launching another unplanned cohort.
