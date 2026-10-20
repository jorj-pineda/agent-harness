# Same-root Gemma read-batching observations — 2026-10-20 UTC

**Completed-and-independently-accepted: 0/2 control,
0/2 treatment.** Artifact acceptance was
0/2 versus 2/2. The intended
batching mechanism did not activate in any available normalized transcript.
Do not ship the experimental sentence or attribute the accepted-patch difference
to batching. The duration workflow remains incomplete under these limits.

The [structured results](../../results/2026-10-20-duodoro-batch-read-comparison.json)
retain all four fixed attempts, durable per-request budget evidence, final grades,
unknowns and audit hashes. The reviewed [protocol](BATCH_READ_COMPARISON.md),
[manifest](batch-read-freeze.json) and [patch](batch-read-treatment.patch) are unchanged.
Original ten-task results (2/10 accepted, 1/10 completed-and-accepted), the earlier
feedback comparison and the release gates retain their original scope and status.

## Every scheduled outcome

Read batches are requested / successful **normalized archived** batches, not a
complete count of raw provider-wire calls. Tokens are reported prompt / output;
time is the saved run-created-to-finished interval, not pure inference or a hard
return deadline.

| Attempt | External artifact | Completion / observed check | Model requests | Read batches | Tokens | Run seconds |
|---|---|---|---:|---:|---:|---:|
| 01-T01-r1-control | Fail | budget_exhausted / not_run | 7 | 0 / 0 | 51,323 / 832 | 172.39 |
| 02-T01-r1-treatment | Pass | budget_exhausted / not_run | 7 | 0 / 0 | 51,648 / 801 | 173.21 |
| 03-T01-r2-treatment | Pass | budget_exhausted / not_run | 7 | 0 / 0 | 51,648 / 801 | 168.03 |
| 04-T01-r2-control | Fail | budget_exhausted / not_run | 7 | 0 / 0 | 51,323 / 832 | 166.40 |

All four stopped at the preventive total-token estimate gate. No observed check
ran and no turn completed; passing external acceptance does not fill in those
missing workflow steps. Both treatment artifacts fix both formatters and preserve
finite-positive/fractional and tag behavior. Both control artifacts fix only
`formatDuration`: an escaped-newline mismatch is
rejected, a later exact edit repairs that first formatter, and the next-request
gate stops before `formatTime` is fixed. External acceptance rejects `NaN:NaN`
instead of `0:00`. Both repeats within each arm produce identical final file bytes.

## Execution and input identity

The inference runtimes were published control `45f4182cde0f7652f7e2f9cad23696b5f7373f11`
and local treatment `1f7e0f6804edccb154ec2542012074f5922a03b7`. Only the compact-prompt
batching sentence differs. Both retain the inherited `coding-v3-compact-v1` label;
the exact commit/tree, prompt hashes and request fingerprints identify the variants.
Production prompts and the reporting runtime are separate from the local treatment.

The actual CLI submitted the original T01 request once per fresh session, in fixed
control/treatment/treatment/control order. All attempts used the same literal
workspace root, frozen full-source Duodoro revision `5324cab`, independent unseeded
stores and checkpoints. The first-request assertion matched each frozen identity
before provider forwarding. Within-arm initial normalized inputs matched across
repeats; across arms only the intended sentence differs. Tool schemas and 2,048
output allowance match. This corrects the earlier cohort's arm-specific root-path
problem, without attesting hidden backend state or deterministic sampling.

Local `gemma4:12b` used frozen digest
`4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`,
Ollama 0.33.2, Python 3.14.5 and Node 24.22.0. Each attempt began unloaded.
The approved 300-second dispatch, 64,000 total / 8,192 output token, 24 tool,
12 iteration, 60-second request, 2,048 request-output and 16,384 context limits
were unchanged. Temperature zero, thinking off, compact/full tools, required
verification, plan gate off, root injection/support off and one bounded completion
retry retain the frozen profile. All instruction and tool-result text was retained.
No warm-up, replacement attempt, manual patch repair, coding hint or extra allowance.

## Mechanism, budgets and client behavior

Manual review of the available archived assistant calls found no response containing
two distinct read-only calls in any attempt. Consequently no requested/successful
read batch or dependent-argument read-batch misuse was observed. This measures
usable normalized history; suppressed or unarchived raw provider calls are not
reconstructed. The batching instruction therefore provides no observed mechanism
explanation for an accepted patch or efficiency gain.

All four complete responses have known usage: **205,942 prompt / 3,266 output
tokens** in total. The numeric durable journals retain every observed request,
response and final token stop without truncation. The structured rows preserve
per-response usage, requested output allowances, heuristic/anchor estimates and
first identities. Every attempt made seven requests with one usable tool call per
assistant response. Control has 11,845 tokens left at its final gate versus 12,197
estimated prompt + 256 minimum output (608 short); treatment has 11,551 left versus
12,251 + 256 (956 short). These are provider reports
and heuristic gates, not independently attested billing or backend compliance.
The recorded upper bounds fit the configured total/output, request-output,
request/tool counts and journal-time caps; they do not prove hard end-to-end timing.

All four CLI submissions reached the client's 120-second response timeout while
server-owned work continued. Read-only actual CLI `runs` and `run --json` inspection
retrieved the original saved outcomes; no coding request was resubmitted. External
acceptance ran separately after settlement. Existing escape-mismatch recovery is
normal configured runtime behavior, not operator repair. No execution setup
correction altered the four submissions; later local report/audit recipes only
process retained evidence. Frozen preparation corrections remain historical.

Two repeats on one selected public task and one backend do not establish a general
causal quality improvement, a held-out result or an efficiency ranking. Equal initial
inputs do not force equal later model choices. Keep the accepted-artifact observation
separate from the zero completed workflows and unobserved batching mechanism.

## Audit, cleanup and disposition

Offline external regrading reproduced all four original artifact grades. The final
audit verified one durable run per attempt, matching saved archives/CLI exports,
frozen settings/criteria/reference/staging bytes and modes, captured profiles and
first identities, clean pinned runtimes/environment and protected-file preservation.
Only `client/src/lib/format.ts` changed in each final artifact. The entire common
workspace was archived after each settled run, API stop and verified model unload;
no artifact was discarded or reused as a later baseline.

Raw stores, transcripts, bounded diffs/checkpoints, final artifacts, CLI/server logs,
operator events and archived recipes remain under
`~/.local/share/agent-harness/trials/duodoro-batch-read-1haw4enm/`.
The structured result embeds the final audit and its SHA-256, plus local recipe
hashes. It is not a portable full-source/full-trace bundle.

The owned API is stopped, its listener absent, and evaluation-loaded Gemma unloaded;
the existing Ollama service remains running. No Duodoro working-checkout edits,
commits/pushes, paid calls, model switches or Docker actions occurred.

**Disposition: retain the production prompt and approved gates.** This prompt-only
batching intervention did not demonstrate completion or activate its intended
mechanism. Next work should select a narrower supported duration workflow or a
separately reviewed context/request intervention with explicit retention and
activation criteria; do not launch another cohort, add features or raise limits
from this result alone. G2's real reversal, G3's ten-task target, G4's held-out/
two-model evidence and G5's release decision remain open/unmet.

This PR is docs/data only. Preparation's two pinned runtimes each passed 1,342
offline tests (five live deselected), static checks and both scripted 90/90
matrices. Those are historical mechanics checks, not newly rerun model-quality
evidence; this report validates saved artifacts and observations independently.
