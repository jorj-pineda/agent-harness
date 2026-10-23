# Ten-task provider-timeout follow-up: protocol

**Preparation only; no inference or execution freeze.** The selected
[timeout comparison](PROVIDER_WAIT_RESULTS.md) completed and accepted both original
T09 attempts at 180 seconds, versus neither control at 60, within unchanged
300-second task allowances. The next question is whether the existing 180-second
provider setting yields useful completion across all ten agreed tasks. A success
on T09 alone cannot answer it.

The [machine-readable plan](provider-wait-ten-task-plan.json) preserves original
requests, order, five CLI/five panel assignments, acceptance/reference/setup hashes,
source-review requirement and reserved recovery limits. This is a new candidate
cohort, not replacements for failed historical cells or a held-out comparison.
There is no production default change or automatic release promotion.

## Fixed scope and resources

Use all ten tasks from the approved [original protocol](TRIAL_PROTOCOL.md) and
[whole-file follow-up](WHOLE_FILE_TEN_TASK_PROTOCOL.md), with their exact original
wording. Do not add successful T09 hints or substitute the formatter recovery
variant for T01. Keep T08's independent final-source reuse review mandatory.

| Task | Client | Adapter | Coding turns |
|---|---|---|---|
| T01 | CLI | duration_format | one |
| T02 | Panel | timer_normalization | one |
| T03 | CLI | calendar_layout | two reserved half-budget phases |
| T04 | Panel | share_storage | one |
| T05 | CLI | timer_prefs | one, followed by post-grade reversal |
| T06 | Panel | shortcut_count | one |
| T07 | CLI | rotation_preview | one, original two-path scope |
| T08 | Panel | duration_refactor | one, independent source review |
| T09 | CLI | tag_filter | one |
| T10 | Panel | wrapup_intentions | one |

Pin local `gemma4:12b` and Duodoro `5324cabba4ad3c8ad2e1cd1480218cf087d9b741`.
Prepare fresh full-source disposable copies and unseeded task-isolated stores.
No paid/new model, dependency installation, service/debug change or Docker.
Do not modify Duodoro's working checkout, index or refs. Applying any eventual
patch would require a separate Duodoro branch/PR.

Use the existing compact/whole_file configuration, temperature zero, thinking off,
16,384 context, 2,048 request output/256 minimum, one completion recovery, identical-
call limit two, two-file edit budget, captures on, root injection/plan gate/support/
inspection off. Verification remains required with exact
`["python", "-m", "pytest", "-q", "test_visible.py"]`.

The candidate changes `request_timeout_seconds` from 60 to **180**. That scalar
changes all four HTTP timeout phases (connect/read/write/pool), not a hard request
deadline. Production buffered HTTP remains. The outer task budget still governs
execution, including T03: its **150-second phase allowance can expire before the
180-second provider timeout**. Do not enlarge, transfer or disable that phase cap.

| Allowance | Ordinary task | Each T03 phase |
|---|---:|---:|
| Dispatch seconds | 300 | 150 |
| Total estimate-gated tokens | 64,000 | 32,000 |
| Output tokens | 8,192 | 4,096 |
| Tools | 24 | 12 |
| Model iterations | 12 | 6 |

At most ten tasks, eleven operator-submitted coding turns and 3,000 reserved
seconds. No unused-budget transfer, third phase or replacement. These are dispatch
allowances, not an end-to-end time estimate; cleanup/review/storage may exceed them.
CLI response inactivity stays 600 seconds; panel uses native SSE. Uncertain
delivery permits original-key GET inspection only, not a new submission or direct
HTTP rescue. No timeout changes after outcomes are observed.

## Evidence and recovery procedure

Use the existing bounded evaluation-only normalized/wire input recorder in every
coding phase, equally: twelve requests, 1 MiB per file, 8 MiB combined, fresh
private plaintext files with owner-only permissions. Do not publish raw payloads.
It adds observation I/O, not backend receipt or chunk timing. Recording failure is
a retained outcome; incomplete evidence blocks further cohort dispatch. Keep all
ten scheduled tasks in the denominator, with undispatched tasks explicitly unrun;
no repair/replacement of the observer or cell after a model submission.

Retain original T03 trigger, exact continuation message, two half-budget phases,
CLI cancel attempt, settlement, identical-profile API restart, saved review and at
most one native guarded fork. The original trigger does not wait for an edit.
Record early/late cancellation, no edit, rejected continuation and unknown usage
honestly. Never substitute a fresh normal session for a blocked fork.

For T05, archive/grade before the original later sentinel mutation and eligible
single-file reversal. Keep sentinel text, request keys and operator procedures
frozen. Preserve later work; rejected/ineligible reversal stays a failure/limit,
not permission to reset. Restart review is read-only and must not replay inference.

Each task gets an unchanged full artifact archive, external grade/regrade, protected
bytes/modes checks, saved result/progress/budget evidence and guarded GET-only
restart review. Stop owned APIs completely; unload evaluation Gemma while preserving
the existing service. Preserve all original artifacts, fixtures and criterion hashes.
Path and command validation are not process isolation; only trusted code is used.

Record artifact acceptance, completed-and-accepted, observed checks, established
known within-budget acceptance, client/recovery outcomes and interventions separately.
Unknown in-flight usage stays unknown. T08 executable success remains provisional
until independent source inspection satisfies its frozen rubric, with reviewer,
criterion/diff hashes and verdict recorded. Pending review is not acceptance.
Passing editable checks, prose, confidence or filename overlap cannot replace it.

## Gates before inference

1. Independent protocol review and merge. The original task/limit agreement and
   standing local Gemma/Duodoro authorization apply; preparation tests do not
   establish execution readiness or approve revised acceptance.
2. Prepare fresh scheduled roots/stores and actual CLI/browser ownership. Recheck
   initial-fail/reference-pass acceptance, exact original request/setup/reference/
   criterion bytes and modes. Never use reference-validation artifacts as starting
   edited workspaces. Keep retained cohorts unchanged.
3. At the clean selected runtime, run exact G0 and both scripted matrices, plus
   representative actual CLI/panel scripted preflight covering the 180-second
   construction, unchanged required checks/captures, T03 phase timeout/cancellation/
   fork and T05 reversal. Exercise the exact source factories, recorder, guards,
   incomplete-evidence stop and default/GET-only restarts without inference.
4. Pin actual runtime/tree/lock/source/assets, executable/PATH/package environment,
   model digest/backend, settings/profiles, roots/identities/full initial manifests,
   absent stores and owner/pending-key state, all observer/launcher/audit recipes
   and receipts. First normalized and exact wire identities need per-task guards
   before forwarding. T03 fork input depends on live history and native guards;
   never fabricate it or bypass those guards for parity.
5. Independently review and merge the immutable actual execution freeze, then
   recheck it unchanged before dispatch. Drift blocks launch; no silent refresh.
   Protocol-only candidate runtime `56406a4` is not an approved execution freeze.

## Decision rules and limits

Keep all ten outcomes, including setup failures and unrun phases. Preserve the
8/10 artifact target, scope/source-review and budget conditions; eight passing
artifacts alone do not establish G3 with unknown usage or missing required review.
If missed, retain no-go. If reached once, report the limited public-task observation;
G2's complete recovery acceptance, repeated/held-out/two-model G4 and Jorge's G5
release decision remain separate. No release tag or package publication.

The historical whole-file cohort remains 2/10 accepted and 2/10 completed-and-
accepted. This new cohort has later runtime observations, recording overhead and
180-second provider settings; it is not an isolated causal timeout comparison.
Do not pool its counts with prior cohorts or two-repeat T09 results. No supported
narrower envelope or general quality gain is assumed. Production adoption requires
a separately scoped decision, even if this candidate cohort succeeds.

## Completed protocol preparation

All ten original initial visible/external checks failed and references passed in
fresh offline copies. Original requests, criterion/reference/setup hashes and setup
modes match. The 28 focused criterion tests passed. The validation recipe, receipt
and unchanged Duodoro head/status/index/refs fingerprints remain at the local root
recorded in the plan. These are deterministic acceptance checks, not inference.

An initial helper used the nonexistent `Task.acceptance_dir` field and stopped
after T01 validation. The corrected helper uses `Task.acceptance` and ran all ten
in a fresh root; the earlier root is retained. No source, criterion or model attempt
was changed. Exact G0 and actual execution-client preflight belong to the next
freeze slice; no full-suite or model-quality claim is made by these protocol checks.
