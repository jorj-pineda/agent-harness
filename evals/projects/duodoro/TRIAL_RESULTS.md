# Frozen Duodoro client trial — 2026-10-19 UTC

**No-go for the frozen unattended task envelope.** The ten-task local Gemma cohort
produced **2/10 accepted artifacts**, below the agreed 8/10 target. Only **1/10
completed and was independently accepted**. No operator repaired a patch, no task
was replaced, and no failed model turn was retried with a fresh allowance.

The [machine-readable observations](../../results/2026-10-19-duodoro-client-trial.json)
retain all ten rows. This is a real-model, real-tool client trial with external
acceptance on final files, not a scripted contract or simulated-tool smoke test.
It is one fixed-order public development cohort, not held-out evidence, a causal
improvement claim, or a same-model baseline comparison.

## Frozen configuration and evidence

Execution used runtime `cd9891718193ae4d9bb56056ae8aea8e57a9ef4c`, the reviewed
[freeze](EXECUTION_FREEZE.md), and manifest SHA-256
`26f9971e635d204599a38968b10427bef967178890a1c041d26eb3cfc7bfb62d`.
The later reporting commits were not used as the model runtime. Duodoro remained
pinned at `5324cabba4ad3c8ad2e1cd1480218cf087d9b741` in disposable full-source
copies; the user's working checkout was not edited or committed/pushed.

Local Ollama `0.33.2` served configured `gemma4:12b`, installed digest
`4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`.
Read-only metadata matched before dispatch; the loaded-model observation matched
that digest during execution. No model was loaded at the initial check. The
managed environment used the concrete Node `24.22.0` binary and the frozen Python
3.14.5 environment, compact/full coding tools, temperature zero, thinking off and
context 16,384. Every task had isolated unseeded storage and turn captures.

Ordinary tasks reserved 300 dispatch seconds, 64,000 total reported tokens,
8,192 output tokens, 24 tools and 12 model iterations. T03 reserved two identical
150-second / 32,000-total / 4,096-output / 12-tool / 6-iteration phases, without
transfer of unused allowance. Requests were capped at 2,048 output tokens and
60 seconds. No settings, prompts or criteria were tuned during the cohort.
These estimate-gated allowances are not a hard end-to-end latency guarantee.

Ten initial requests used their assigned real clients: five CLI and five panel.
The one additional submission was T03's frozen, guarded continuation. Reference
patches were not supplied to the model. The operator provided no coding hints,
manual patch repair or unplanned model follow-up. All authoritative acceptance
checks ran outside the editable roots using the pinned owned bootstrap.

## All ten outcomes

| Task | Client | Final coding outcome | Observed check | External artifact acceptance | Observed blocker |
|---|---|---|---|---|---|
| T01 duration formatting | CLI | budget_exhausted | failed | Fail | Two exact-edit mismatches, then total-token gate; final files unchanged. |
| T02 parser flooring | Panel | budget_exhausted | not_run | **Pass** | Correct bounded parser edit; token gate stopped verification/completion. |
| T03 calendar/recovery | CLI | cancelled → budget_exhausted | not_run | Fail | Guarded second phase exhausted its reserved token allowance without a patch. |
| T04 share storage | Panel | budget_exhausted | not_run | Fail | Repeated exact-edit mismatches, then token gate; final files unchanged. |
| T05 timer preferences/reversal | CLI | failed run; final response unavailable | unavailable | Fail | Confirmed provider read timeout; no patch; planned reversal request rejected. |
| T06 shortcut count | Panel | failed run; final response unavailable | unavailable | Fail | ProviderError; underlying cause not retained in stream log; no patch. |
| T07 rotation preview | CLI | failed run; final response unavailable | unavailable | Fail | Confirmed provider read timeout; no patch. |
| T08 duration refactor | Panel | failed run; final response unavailable | unavailable | Fail | ProviderError; underlying cause not retained in stream log; no patch. |
| T09 tag filter | CLI | budget_exhausted | not_run | Fail | Exact-edit mismatches, re-reading and token gate; final files unchanged. |
| T10 wrap-up intentions | Panel | **completed** | **passed** | **Pass** | No remaining failure observed within the library criterion scope. |

The final task statuses were five budget stops, four failed runs with unavailable
normal response evidence, and one completed task. T03's planned cancelled source
is a separate phase, not an eleventh task. Visible checks passed once, failed once,
were recorded as not_run in four final outcomes, and were unavailable for four
failed runs. Missing fields were not converted to clean/not-run outcomes.

T02 and T10 passed the frozen behavioral and scope criteria. Their bounded diffs
were reviewed: T02 floors the final clamped duration expressions; T10 trims,
filters and deduplicates the eligible intentions. T02 is an accepted artifact,
**not** a completed-and-checked task. T08's executable criteria failed because no
helper patch existed; independent refactor review was not performed or reported
as passing. An unused helper would still not satisfy its frozen contract.

| Category | Attempts | Accepted artifacts | Completed and accepted |
|---|---:|---:|---:|
| Bugfix | 6 | 2 | 1 |
| Feature | 3 | 0 | 0 |
| Refactor | 1 | 0 | 0 |

## Client recovery, preservation and interventions

All four ordinary CLI chat tasks (T01/T05/T07/T09) reached the client's
120-second response timeout while server-owned work continued. The operator
retained the printed request keys and inspected saved runs through the CLI;
no new coding submission was made. Panel failures T06/T08 were inspected with
the panel's Check run control, which displayed a failed lifecycle and read-only
review, without inventing a normal completion/check response. No direct HTTP
workaround replaced a client control. Raw-record CLI exports of panel runs were
independent evidence collection, not replacement panel submissions or rescue.

T03 followed the planned single keyed running lookup and one cancellation.
Cancellation arrived before the first completed model response, so it preserved
an unchanged workspace rather than a partial patch. After owned settlement and
an actual API process restart with identical storage/settings, CLI reviews showed
unchanged workspace/modes, valid transcript and matching configuration. The
pinned latest turn plus explicit scope acknowledgment admitted exactly one fork.
The source run JSON remained identical afterward. This demonstrates the exercised
controls, not successful coding recovery after a real partial edit.

For T05 the artifact and failing grade were preserved first. The operator then
appended the frozen later sentinel text and submitted the one planned single-file
reversal request. It was rejected; no saved reversal action was found before or
after restart. The failed run had no after capture. The operator did not retry,
force a restore, reset Git or alter the target. Final manifest comparison showed
the target bytes/mode and all other files at baseline, with only the intended
later sentinel edit retained. **A successful real-trial reversal was not
demonstrated.** The earlier deterministic preflight's success remains separate.

All ten external scope checks and final audits found no non-allowed changes,
with the deliberate post-grade T05 sentinel exception handled separately. This
covers the frozen regular-file/content scope, not ignored contents, external
writers, ownership/ACLs or host process isolation. No Duodoro PR was created:
these were disposable evaluation patches, not requested product changes.

## Budgets, timing and diagnosis limits

Six phase reports had known usage; five were unknown (the cancelled source plus
four failed runs). Known phase reports sum to 274,169 prompt and 6,206 output
tokens, **not a full cohort total**. Additional usage can exist for interrupted
requests and unavailable failed-run prefixes; no zero or estimated replacement
was inserted. Five tasks had known phase usage and observed journal intervals
within their frozen allowances, including both accepted artifacts; five tasks'
full budget compliance remains unknown. A budget stop can occur below the nominal
token cap when another complete request cannot fit.

The eleven created-to-finished run journal intervals sum to approximately
1,683 seconds (28.0 minutes), including setup/final review/storage. This excludes
operator gaps and grading, is not pure inference time, and is not a hard dispatch
or return-time measurement. The 300-second allowance does not bound all cleanup
and review work.

Saved tool arguments for T01/T04/T09 show escaped newline sequences and rejected
exact matches. That is a concrete edit-interface failure observation; the cohort
does not establish that any particular prompt/tool change would repair it.
T05/T07 logs explicitly contain ReadTimeout. The stream path logs only
ProviderError for T06/T08, so their underlying causes are unknown. Failed runs
also lacked normal final usage/check responses. The recorded request limit was
60 seconds, but without complete per-request generation timing this does not
establish why generation or transport exceeded it. No timeout extension, model
switch, larger budget or retry was introduced to improve this cohort's score.

## Validation, cleanup and release disposition

The final audit rehashed every criterion, reference, settings file and baseline;
the frozen manifest and clean runtime matched their pins. All captured runtime
profiles matched the frozen settings, prompt/tool hashes and provider options.
All ten saved coding artifacts reproduced their original external grades, including
T05's artifact saved before its later preservation action. The machine-readable
result retains raw evidence hashes and unknown fields; full local stores,
transcripts, CLI output, artifacts, audit logs and operator scripts remain under
`~/.local/share/agent-harness/trials/duodoro-cd98917-freeze-ifjtk9jq/`.
Raw source/captures remain local; this report is not a portable full-trace bundle.

The temporary API listener was released. The evaluation-loaded Gemma runner was
unloaded with `ollama stop gemma4:12b`; `/api/ps` showed no loaded models afterward.
The pre-existing Ollama service was left running. No Docker or paid calls occurred.

No application/evaluator implementation or criteria changed in this reporting PR.
Its checks cover result consistency and artifact reproduction. G0's 1,291 offline
tests, static checks and two scripted 90/90 matrices belong to the frozen runtime
validation, not a newly rerun suite or model-quality evidence for this report.

G1 agreement/freeze review was completed before inference. G2 remains open:
client submission/review and cancel/restart/fork were exercised, but real reversal
was not demonstrated. G3's 8/10 artifact target was not met. G4's held-out/repeated
two-model comparison and G5's release decision remain unmet. Do not release this
configuration as an unattended coding assistant for the frozen task envelope.
The next useful decision is whether to narrow that envelope or investigate the
recorded edit/timeout failures with a controlled follow-up—not another unmeasured
feature batch. This cohort makes no general claim that Gemma cannot code.


Subsequent offline investigation: [exact-edit mismatch diagnosis](EDIT_MISMATCH_DIAGNOSIS.md)
reproduces nine dispatched failures and extends specific rejection feedback.
It does not rerun or revise this cohort, its criteria, or its acceptance counts.
