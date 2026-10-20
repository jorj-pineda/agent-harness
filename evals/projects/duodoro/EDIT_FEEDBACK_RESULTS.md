# Paired Gemma edit-feedback observations — 2026-10-20 UTC

The frozen twelve-attempt development comparison produced **0/6
accepted artifacts with control and 6/6 with treatment**.
Completed-and-accepted was **0/6 versus
4/6**. Acceptance, completion and observed checks
remain separate. These are observed outcomes for selected development tasks, not
held-out evidence or proof of a general causal improvement.

The [structured observations](../../results/2026-10-20-gemma-edit-feedback-comparison.json)
retain every scheduled attempt, both repeats, raw evidence hashes and unknowns.
The [reviewed freeze](EDIT_FEEDBACK_COMPARISON.md) and [manifest](edit-feedback-comparison-freeze.json)
remain unchanged. This cohort does not replace the original ten-task trial's 2/10
accepted and 1/10 completed-and-accepted results or meet the release gates.

## All paired outcomes

Each cell shows external artifact acceptance / completion / observed check.

| Task | Repeat | Control | Treatment |
|---|---:|---|---|
| T01 | 1 | Fail / budget_exhausted / not_run | Pass / budget_exhausted / not_run |
| T01 | 2 | Fail / blocked / not_run | Pass / budget_exhausted / not_run |
| T04 | 1 | Fail / budget_exhausted / not_run | Pass / completed / passed |
| T04 | 2 | Fail / budget_exhausted / not_run | Pass / completed / passed |
| T09 | 1 | Fail / budget_exhausted / not_run | Pass / completed / passed |
| T09 | 2 | Fail / budget_exhausted / not_run | Pass / completed / passed |

All twelve initial requests used their frozen real clients: eight CLI and four
panel, in the counterbalanced order. Each used a fresh disposable full-source
copy and separate unseeded storage. No task was replaced, no artifact was repaired
by the operator, and no failed coding submission received a new key or allowance.
Normal bounded harness recovery retained the frozen settings; no operator coding
hint or unplanned model follow-up was supplied.

## What was exercised

Both runtimes use base `5974d2201cb12829de2b593b602c649515e98b0d`, including safe
provider-failure metadata. Treatment is that published revision; control is local
experiment commit `1c9249af2b5bcaad2e1628c130a9e879c934196f`. Only the executable
mismatch helper differs. The reporting checkout/merge revision was not used as
the model runtime. The original requests, protected baselines, independent criteria,
references, visible commands and settings remained frozen; no editable test could
redefine acceptance.

Local `gemma4:12b` used the frozen digest and Ollama 0.33.2, concrete Node 24.22.0
and Python 3.14.5, compact/full tools, temperature zero, thinking off and context
16,384. Every attempt began with the model unloaded. Limits stayed at 300 dispatch
seconds, 64,000 total reported tokens, 8,192 output tokens, 24 tools, 12 iterations,
60 seconds per request and 2,048 output tokens per request. Setup/cleanup/review
are not a hard return-time guarantee. No inference warm-up, new model, timeout
extension, paid call or premium-model coding supervision occurred.

Control's rejected calls remained rejected. Treatment's specific feedback also
rejected the original malformed arguments; the model had to submit later edits.
The structured rows retain mismatch errors and successful exact edits separately.
A later successful edit does not by itself establish that the feedback caused it.
Tool-call record counts include rejected/not-dispatched records and are not an
independent count of tool execution.

## Attribution and scope limits

**This is not a clean single-variable contrast at the complete request level.**
The executable runtime difference is isolated, and captured base-prompt/schema/
settings profiles match. However, the shared runtime appends each absolute
workspace path to the system prompt. Those frozen paths differ per attempt and
include arm names. The saved rendered-system hashes and hashes after replacing
the workspace path document that distinction; equal base-profile hashes do not
mean identical rendered model inputs. All twelve saved system hashes differ;
all twelve match after replacing the exact workspace path with `<WORKSPACE>`.

Tool-call prefixes through the first mismatch diverged in 2 of
5 pairs with comparable evidence; unknown or absent first mismatches
remain unknown. This metric compares tool names/arguments, not complete wire
requests or hidden model state. Temperature zero did not establish deterministic
replication. No outcomes were retuned after observing divergence.

The three public library tasks were chosen because earlier attempts failed on
escaping. Two repeats per arm/task, cold-start policy, fixed order and one local
model/backend do not establish held-out success, a model ranking, full-app coverage
or a general quality/efficiency gain. Scope checks cover the frozen regular-file/
content/mode policy, excluding ignored contents and host process isolation.

The second T01 control stopped on repeated grep before attempting an edit, so
that pair did not exercise the mismatch diagnostic in both arms. It remains in
the fixed denominator; its outcome cannot demonstrate an effect of the new hint.

## Client behavior, usage and interventions

7 CLI chat submissions reached the client's 120-second response timeout.
The server-owned run continued; saved identities were inspected through real CLI
lookup/review commands without another coding submission. Panel tasks used actual
session/capture/send controls and rendered final completion/check evidence.
Raw-record CLI exports of panel runs were independent collection, not a substitute
client coding submission or a manual HTTP rescue.

All twelve attempts have complete reported usage: **616,347 prompt and 14,454
output tokens**. Control accounts for 326,503 / 6,981 and treatment for
289,844 / 7,473 prompt/output tokens. These are provider reports, not independently
attested backend accounting or an efficiency claim. No provider failure was
recorded. Missing checks were never filled with success. Timing scopes are
run-created-to-finished journal intervals or reported loop latency, not pure
inference or hard end-to-end dispatch/return measurements. Journal intervals range
from 86.42 to 230.05 seconds. All twelve have sufficient observed upper bounds for
the recorded total/output/time/iteration/tool caps; this does not attest every
per-request backend limit. Two treatment artifacts passed despite a total-token
estimate stop before verification/completion. Control had five budget stops and
one repeated-grep block.

Operator setup corrections are journaled: the first lookup used unsupported
`runs --json`, which failed locally before HTTP and was replaced by `runs` then
`run --json`; the first unload CLI lacked HOME and panicked locally. The API had
stopped, the next-start cold guard blocked, and adding HOME to the cleanup-only
CLI environment allowed owned Gemma unloading. No second coding attempt started
until the unloaded state was verified. Neither correction changed inference
settings, artifacts, model inputs or scheduled attempt counts. A later pre-start
port-release guard briefly blocked attempt 10 before API/session/model submission;
listener/bind/HTTP checks then confirmed the port free and the untouched attempt
started. The cleanup helper subsequently waited for bind readiness before declaring
settlement. A subsequent cleanup check after attempt 10 reproduced a plain-bind
failure while a reusable bind succeeded with no listener and an HTTP ConnectError.
The operator probe was corrected to check TCP listener absence and server-equivalent
reusable binding. The owned model was unloaded before attempt 11. These setup-only
corrections did not add or repeat a coding submission.

## Validation and disposition

The final audit reproduced every saved artifact's original external grade and
verified one durable run per attempt, clean pinned runtimes, captured profile/
settings/criterion hashes, unchanged protected files and final cleanup. Raw stores,
transcripts, final artifacts, client/server evidence and operator scripts remain
under `~/.local/share/agent-harness/trials/duodoro-edit-feedback-sgpr8sdv/`.
The report is not a portable full-source/full-trace bundle.

The owned API listener was released and evaluation-loaded Gemma unloaded. The
existing Ollama service was left running. No Duodoro working-checkout edits,
commits/pushes, paid calls or Docker actions occurred. The reporting PR changes
only observations/documentation; pinned offline gates are historical mechanics
evidence from preparation, not newly rerun model-quality validation. G2's real
reversal, the broader ten-task target and held-out/two-model release requirements
remain unmet. A next measurement needs explicit attention to rendered workspace
identity and the remaining timeout/completion failures; do not infer release
readiness from this selected-task contrast.
