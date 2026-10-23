# Release-readiness audit — 2026-10-18

Audited source: `62ac6a44f1a6d96e7ea261162d51bc662306cefa` (merged through PR #143).
This is a point-in-time source, regression and recorded-evidence audit. It is not
an independent security audit, a fresh model comparison or release approval.

**Decision at the original audited revision: suitable for a controlled, supervised trial from a source checkout;
not yet ready to claim the roadmap's personal-use release targets.** Infrastructure
is substantially implemented. Reliable coding performance and the complete real
workflow are still unproven. PR/test counts are not a release-completion percentage.

## Current disposition — 2026-10-23

**Retain no-go for unattended coding under the agreed ten-task envelope.** The
later [provider-timeout comparison](evals/projects/duodoro/PROVIDER_WAIT_RESULTS.md)
completed and accepted original T09 at 180 seconds in 2/2 attempts, versus 0/2 at
60 seconds, within unchanged 300-second task limits. This selected public-task
result supports broader candidate validation, not a production default or G3/G4
closure. The [ten-task follow-up protocol](evals/projects/duodoro/PROVIDER_WAIT_TEN_TASK_PROTOCOL.md)
preserves original tasks, clients, criteria and 300/150-second budgets; fresh
client preflight, exact-runtime G0 and independent execution-freeze review remain
required. No inference in that preparation slice or release approval follows.

The earlier [transport comparison](evals/projects/duodoro/TRANSPORT_COMPARISON_RESULTS.md)
finished buffered **0/2 accepted, 0/2 completed-and-accepted** and streamed
**0/2 accepted, 0/2 completed-and-accepted** on original T09. All four fifth calls
received zero body chunks before a roughly sixty-second read timeout, with no
edits. Earlier treatment responses contained valid streamed frames, so the
experimental path was active; it did not remove the observed failure. Backend
cause and failed-call usage remain unknown. Two repeats per arm on one public
task do not establish a general transport effect or coding-quality comparison.

The streaming-only remedy hypothesis is unsuccessful **for this tested task and
configuration**. Close this comparison without promoting the transport wrapper to
application settings/defaults. Keep its code, freeze and raw results as evaluation
reproduction evidence. The production adapter still requests buffered responses;
original prompts, limits and completion/recovery rules remain unchanged.

The original [ten-task trial](evals/projects/duodoro/TRIAL_RESULTS.md) accepted
2/10 artifacts and completed-and-accepted 1/10. The later
[whole-file ten-task cohort](evals/projects/duodoro/WHOLE_FILE_TEN_TASK_RESULTS.md)
accepted 2/10 and completed-and-accepted 2/10, again below the unchanged 8/10 target.
Keep these cohorts separate from selected-task comparisons and workflow probes;
additional PRs, green offline tests and successful bookkeeping do not close their
coding-quality acceptance gap. No narrower unattended envelope is established.

| Gate | Current evidence | Work still required |
|---|---|---|
| G0 — mechanics baseline | Passed at the latest frozen execution runtime `6721cc9`: 1,453 offline tests, static checks and both scripted matrices. | Validate the exact next execution/release candidate; this is not a model-quality gate. |
| G1 — trial agreement | Original ten tasks, limits and criteria agreed and reviewed before inference. | Any revised envelope must be named and frozen explicitly; the original agreement does not make new criteria equivalent. |
| G2 — client/recovery workflow | Original client cohort plus one bounded [CLI](evals/projects/duodoro/RECOVERY_RESULTS.md) and one [panel](evals/projects/duodoro/PANEL_RECOVERY_RESULTS.md) partial-edit/reversal observation. | Broader agreed client/recovery acceptance remains open; selected variants and unknown interrupted usage do not close it. |
| G3 — useful coding envelope | Both ten-task cohorts miss the 8/10 artifact target. | Demonstrate a useful agreed envelope within its budgets, or explicitly revise the proposed scope with its unmet original target retained. |
| G4 — repeatable quality evidence | Selected public-task observations exist; no completed matched held-out two-model release comparison. | Expand the 20–30-task/category coverage, preserve held-out cases, and run repeated comparable intended-model measurements. |
| G5 — release decision/onboarding | [Packaged assets and fresh installations](WHEEL_INSTALLATION.md) have bounded evidence; release acceptance is unmet. | Reconcile the open gates, validate the selected distribution/candidate and obtain Jorge's release decision. No release is approved here. |

### Boundary for the next work

This closes the three-slice transport preparation/results/disposition batch. The
current default remains a trusted-local experimental harness for controlled use;
it is not a validated unattended coding envelope. A further investigation should
start with a concrete hypothesis tied to retained evidence, or an explicit proposal
for a narrower task envelope. A fresh live experiment needs its own fixed scope,
actual inputs, independent criteria, comparable budgets and reviewed execution
freeze; failed cells in this completed cohort are not replay slots.

The zero-chunk observations alone do not distinguish backend queueing, generation,
serialization or another cause. No further runtime remedy follows from them yet.
Keep feature batches frozen and the original G2–G5 requirements visible. This
disposition adds no inference, limit increase, model switch, Duodoro patch or
milestone-completion claim. Next development must remove an observed obstacle or
supply missing acceptance evidence, rather than extend the PR count.

## Freeze the trial scope

- One local user, one API process, one active writer per workspace; trusted
  repositories on the tested POSIX/macOS/Linux paths. Serve on loopback.
  `user_id` is an owner convention, not authentication or authorization for a
  shared network service.
- Start from a source checkout using [demo.md](demo.md), explicit settings and the
  already authorized local `gemma4:12b`. No premium supervisor or paid fallback.
- Bounded bugfixes, small features and focused refactors. Inspection-only questions
  are available, but their factual review is separate from patch acceptance.
- Real model work uses disposable copies of the authorized pinned Duodoro source
  or trusted fixtures. Never use Jorge's working checkout as an evaluation fixture.
  Any actual Duodoro source change goes through its own branch and PR, never a
  direct push to main. A copied patch is not automatically approved for application.
- Explicit cancellation, saved review, guarded continuation forks and eligible
  single-file reversal. No automatic resume, arbitrary model switching or atomic
  whole-workspace revert.

This trial scope does **not** lower the roadmap's existing release criteria or
mark Milestones 2–6 complete. No release tag/package publication is authorized by
this audit. Features outside this scope stay frozen unless a reproducible trial
failure makes one necessary.

## What is implemented, and what that evidence means

“Regression covered” below means deterministic/scripted-provider tests, sometimes
with real files/processes. It does not mean a model reliably uses the feature.
Links identify the implementation and representative regressions inspected.

| Capability | Audited status and evidence | Remaining boundary |
|---|---|---|
| Shared runtime/providers | Implemented: [runtime](harness/runtime.py), [provider construction](harness/providers.py), [shared-runtime tests](tests/test_harness_runtime.py). | No claim that every backend obeys requested generation limits. |
| Read/search/exact edits | Implemented: [code tools](tools/code.py), [tool tests](tests/test_tools_code.py), [integration tests](tests/test_tools_code_integration.py). | Path/byte limits and stale-content checks are not OS isolation or an external-writer lock. |
| Managed commands | Regression covered: [process execution](tools/process.py), [timeout/cancellation/environment tests](tests/test_tools_process.py). | POSIX process-group cleanup; synchronous workers and remote inference may outlive a requested deadline. Trusted code still has host permissions. |
| Pre-action controls | Implemented: [loop](harness/loop.py), [policy](harness/policy.py), [context budgets](harness/context.py), [loop regressions](tests/test_harness_loop.py). | File-count/plan gates apply to file-edit tools, not arbitrary writes inside repository commands. Token gates estimate; setup/cleanup/review/storage are not a hard return-time guarantee. Explicit trial budgets remain to be agreed. |
| Completion/check reporting | Regression covered: [outcomes](harness/outcome.py), [latest-check/staleness tests](tests/test_harness_outcome.py), [completion recovery](tests/test_harness_loop.py). | Relevant checks and file-tool edit ordering are observations; a passing editable check cannot establish independent patch acceptance or final-answer truth. |
| Actual workspace review | Implemented: [runtime snapshots/diffs](harness/runtime.py), [dirty-start/index preservation tests](tests/test_harness_runtime.py). | Bounded and excludes paths/content; external writes cannot be attributed. Unavailable review cannot be treated as an unchanged workspace. |
| Durable review and submission keys | Regression covered: [session store](harness/session_store.py), [restart archives](tests/api/test_session_persistence.py), [idempotent runs](tests/api/test_runs.py). | Final responses, run identity and bounded progress metadata are saved; not a full durable live-event/output stream or an automatic replay checkpoint. |
| Cancel/disconnect/shutdown | Regression covered: [API ownership/lifecycle](api/server.py), [cancellation tests](tests/api/test_cancellation.py). | Disconnect leaves app-owned work running under budgets; explicit cancel requests cleanup. Live real-project/client recovery remains unproven. |
| Guarded continuation | Implemented fork, not in-place restart: [guards](harness/continuation.py), [restart/drift/bypass tests](tests/api/test_continuation.py). | Requires current scoped workspace/configuration/transcript/context evidence. Unknown/changed profiles block; uncaptured dependency/external state remains the owner's responsibility. |
| Checkpoints/reversal | Regression covered: [captures](harness/checkpoints.py), [reversal](harness/reverts.py), [preservation tests](tests/test_file_revert.py), [durable reversal tests](tests/api/test_file_revert.py). | Explicit opt-in plaintext storage; eligible single path only, POSIX requirements, external races and excluded metadata/content. No automatic dirty reset. |
| Profiles/instructions/memory | Implemented bounded startup selection and review: [profiles](harness/model_profiles.py), [root instructions](harness/repository_instructions.py), [memory store](memory/store.py), [profile tests](tests/test_model_profiles.py). | Not automatic switching or complete generation/tool-capability negotiation. Instructions are root-only opt-in; notes are observations, not authority. |
| Inspection sessions and clients | Implemented through #143: [tool policy](harness/access.py), [HTTP/restart/fork tests](tests/api/test_inspection.py), [CLI compatibility tests](tests/test_cli_inspection.py), [panel](ui/app.js). | Restricts model tools, not explicit owner actions or host processes. Clients require true acknowledgment before inspection chat; saved history remains review-only. |
| Context management | Estimate gates and pressure diagnostics exist: [budget](harness/context.py), [configuration review](harness/configuration_review.py). | No retained compaction strategy. Context exhaustion stops honestly; long-task continuity is not supported evidence. |
| Acceptance evaluation | Independent checks/disposable workspaces exist: [real runner](evals/real_run.py), [acceptance-boundary tests](tests/evals/test_acceptance_boundary.py). | Visible checks are editable; external checks remain authoritative. Owned pytest startup is not hostile-code isolation. Development tasks are not a held-out benchmark. |

## Recorded model evidence: enough to identify problems, not declare success

See [BASELINES.md](evals/BASELINES.md) and its pinned raw reports/settings. Historical
results keep their original runtime, tool names, grading policy and limitations;
they are not reruns of the audited revision.

- The public Python development catalog has **nine tasks** in
  [tasks.yaml](evals/real_tasks/tasks.yaml): bugfixes, two features and a refactor.
  It is short of the 20–30-task target and has no repository-question category.
- [Duodoro preparation](evals/projects/duodoro/README.md) defines **three pinned
  library tasks**. The calendar task has no recorded inference there. These do not
  constitute ten completed end-to-end client workflows or full application checks.
- The October 13 free-Go comparison on two Duodoro tuning tasks recorded compact
  **4/6 accepted artifacts but only 1/6 completed turns**. It was not held out or a
  current pricing/availability promise; Gemma and Go historical trials are not a
  repeated matched two-model release comparison.
- The October 18 Gemma eight/twelve-iteration trial accepted **2/3 artifacts at each
  cap**, with completed-and-accepted **0/3 versus 2/3**. Both feature artifacts
  failed external long-leading-zero cases despite passing visible checks. One
  twelve-cap refactor finished within eight responses, so extra iterations do not
  explain its completion difference. One sample per task/cap establishes no
  repeatable improvement.
- The previous bounded read-context trial was stopped and retired. No successful
  long-context compaction strategy or held-out improvement is established.

Do not aggregate these unlike cohorts into one success rate. Independent artifact
acceptance, completion, observed check state and human intervention are separate
columns. A legitimately preserved unrelated failing check is not automatically an
incorrect patch; a passed visible check is not automatically a correct one.

## Finite release gates

Dispositions in this table are the historical audit at `62ac6a4`. See
[Trial preparation follow-up](#trial-preparation-follow-up) for subsequent G0/G1
progress; the acceptance requirements remain unchanged.

| Gate | Disposition at audited revision | Evidence required to close it |
|---|---|---|
| G0 — source/regression baseline | Audit checks recorded below; mechanics covered within stated limits. | Keep the gate green at the exact trial/release revision. Investigate any preservation, duplicate-edit or false-status regression before live work. |
| G1 — frozen trial agreement | Open; needs Jorge's choices, not another feature train. | Ten named, bounded real coding tasks; immutable acceptance/scope checks; chosen profile, per-task time/token limits, allowed clarification and rescue rules. Duodoro/local Gemma are already authorized; no new repository/model/paid budget is assumed. |
| G2 — real client/recovery workflow | Open. | Ten distinct agreed tasks through CLI/panel, with raw evidence, every rescue and failures retained. Demonstrate cancel → server restart → review → guarded continuation, and explicit single-file reversal preserving pre-existing/later unrelated work. No manual HTTP calls to make a client workflow work. |
| G3 — useful coding envelope | Open. | Roadmap target: at least 8/10 acceptable task artifacts without manual patch repair, within agreed budgets; record workflow completion/checks separately. Failed/blocked/over-budget cases remain visible. If missed, narrow/revise the proposed envelope explicitly or report no-go. |
| G4 — repeatable quality evidence | Open. | Existing roadmap calls for 20–30 tasks, held-out cases and repeated comparable runs on at least two intended models; demonstrate improvement over the same-model baseline with variation/raw counts. An intended second model has historical observations, not this completed comparison. Recheck free-provider availability/limits before any authorized replay; no silent paid substitution. |
| G5 — release decision and onboarding | Open. | Reconcile G1–G4, rerun preservation/adversarial status checks, verify clean setup in the supported distribution, document exact supported tasks/settings and unmet limits. Jorge decides whether the evidence meets the unchanged release criteria or explicitly revises them. Passing G0 alone is not release approval. |

A controlled trial can start after G0/G1 without claiming G2–G5 are satisfied.
Compaction is required before claiming the Milestone 4 long-task acceptance, but
is **not automatically a prerequisite for a short bounded trial** that stops
honestly on context exhaustion. Build it only if the chosen envelope requires it
or a frozen trial failure demonstrates the need; record that limitation in either
case. Do not erase the full-roadmap criterion by narrowing the trial.

### Proposed execution order, not authorization to run the trial yet

1. Review this audit and freeze feature additions. Correct documentation ambiguity
   now; retain existing limits explicitly rather than opening more feature batches.
2. Prepare the ten-task ledger and frozen acceptance before inference. Use ten
   coding tasks, not ten repeats of one task. Proposed coverage is five via CLI
   and five via panel, spanning bugfix/feature/refactor with the recovery sequences
   above. Ask Jorge to agree tasks and budgets before dependent live work.
3. Run all agreed tasks in disposable copies at one recorded revision/settings
   cohort. Include setup failures, context/time stops and every rescue. Do not
   silently replace unsuccessful tasks or rerun until a favorable count appears.
   Inspection questions can have a separate factual rubric; do not substitute
   them into an “8/10 acceptable changes” denominator.
4. For each task record pinned source/criterion hashes, client, model/profile,
   configured limits, request/run IDs, acceptance, completion/checks, changed paths,
   known/unknown usage, timings and manual interventions. Score the final coding
   artifact before an explicitly requested discard/reversal; assess preservation
   after reversal separately. Full workflow failures remain visible.
5. Review the ledger. Fix only reproduced blockers, in focused PRs with regression
   evidence. Retain original failed attempts; label any later cohort with its new
   revision/settings. Complete the separate held-out/two-model requirements before
   claiming the full release targets; otherwise make an explicit scope decision.

## Trial preparation follow-up

The [ten-task Duodoro protocol](evals/projects/duodoro/TRIAL_PROTOCOL.md) records
Jorge's approval of the tasks, 300-second dispatch allowance, token/profile limits
and reserved recovery phases. All ten acceptance sets now validate offline;
seven new adapters include independent external criteria and reference patches.
T08 also requires independent review of the final refactor's actual reuse.
The ledger remains unrun. The [deterministic client preflight](evals/projects/duodoro/CLIENT_PREFLIGHT.md)
passed the representative exercised paths. The [execution freeze](evals/projects/duodoro/EXECUTION_FREEZE.md)
now pins runtime `cd98917` and records G0 there with the concrete Node 24 toolchain.
G1 agreement and freeze review completed before authorized inference, following
PR #148's merge. The [completed trial](evals/projects/duodoro/TRIAL_RESULTS.md)
retains all ten outcomes: **2/10 accepted artifacts; 1/10 completed-and-accepted**.
The committed ledger remains an unrun template; observed results are separate.
G2 remains open: cancel/restart/fork controls ran, but cancellation preceded a
partial edit and the planned real reversal was rejected. G3's 8/10 target was
missed. G4/G5 remain unmet. Current decision: **no-go for the frozen unattended
task envelope**; choose a narrower envelope or controlled failure investigation.
This follow-up preserves the point-in-time audit table above. No implementation
or acceptance criteria changed, and no quality gain is claimed.

## Selected feedback comparison follow-up — 2026-10-20 UTC

The reviewed [twelve-attempt comparison](evals/projects/duodoro/EDIT_FEEDBACK_RESULTS.md)
observed control **0/6 accepted, 0/6 completed-and-accepted** versus treatment
**6/6 accepted, 4/6 completed-and-accepted** on three selected public tasks.
All final artifact grades reproduced and protected-file checks passed. Differing
rendered workspace paths with arm names, pre-feedback divergence and one pre-edit
control block limit attribution. The original ten-task counts, no-go decision and
G2–G5 remain unchanged. This comparison neither demonstrates real reversal nor
supplies held-out/two-model evidence. No release approval follows from it.

## Packaged panel follow-up — 2026-10-20

The reproduced missing-asset blocker is fixed: wheel builds include `ui/` at
`api/ui/`, and the API resolves packaged assets before the source-checkout fallback.
The offline [packaging regression](tests/test_wheel_panel.py) builds the source
distribution and wheel, installs the wheel with `--no-deps` into temporary storage,
then imports its API with isolated Python outside the checkout. All four asset
bytes and panel/static routes match; API routes retain precedence, and coding-only
startup opens temporary stores without Chroma or inference. The regression failed
on the missing wheel entry before the fix.

This reuses the installed dependency environment; it does not establish a fresh
dependency installation, a complete OS/Python compatibility matrix, release
approval or a model-quality improvement. G2–G5 remain open. The point-in-time
finding and reproduction below describe the original audited revision.
Eight focused regressions and 1,343 offline tests passed (five live deselected),
plus Ruff, formatting and mypy. The existing Chroma deprecation warning remains;
the installed coding-only smoke itself does not import Chroma. No evaluation
behavior changed, so the scripted matrices were not rerun locally for this fix.

## Fresh installation follow-up — 2026-10-20

At `326e69c`, separate Python 3.11.15 and 3.14.5 environments on macOS arm64
installed the built wheel and its lock-constrained dependencies without system
site packages. Both passed dependency checks, actual console/API startup, panel
asset serving, session/manual-checkpoint/configuration inspection and checkpoint
review after restart. Zero coding runs/inference; fixture bytes/mode and shutdown
were preserved. [Installation evidence](WHEEL_INSTALLATION.md) records cache
misses, registry retries with unchanged constraints, a corrected operator shutdown
assertion and retained hashes/logs. This supplies the narrow fresh-install evidence
missing from the packaging regression, not unconstrained-dependency compatibility,
interactive browser/model acceptance, G2 recovery or G5 release approval. Original
trial outcomes and G2–G5 remain unchanged. Docs/data only; no new full gate run.

## Optional support dependency follow-up — 2026-10-20

Base installs now omit Chroma; `[support]` explicitly enables its installation,
while `[dev]` keeps the full regression environment unchanged. A fresh
lock-constrained macOS/Python 3.11.15 observation at `34976b8` records 38 base
distributions versus 86 after adding support, normal resource lifecycle in both
modes and a pre-resource missing-extra error. Both installed-base scripted matrices
pass 90/90 with Chroma absent. [Evidence and limits](WHEEL_INSTALLATION.md#optional-support-dependencies)
remain separate from coding-quality/release evidence; no inference or retrieval
request ran. Prior install observations retain their original runtime/counts.

## Existing editing-preset observations — 2026-10-20

The [same-root T01 comparison](evals/projects/duodoro/TOOLSET_RESULTS.md) at
`ce5d54c` observed full **0/2 accepted, 0/2 completed-and-accepted** versus
whole-file **2/2 accepted, 2/2 completed-and-accepted**, with passing checks.
Both successful answers also include an unexecuted inaccurate test illustration;
artifact acceptance does not validate all prose. Regrades and preservation/evidence
audits passed; all owned APIs/model were stopped/unloaded. No operator coding
repair, replacement or default switch occurred. One selected public task/two
repeats and whole-file stale-hash/large-write risks do not meet G3/G4 or approve
release. Original cohorts/no-go and G2–G5 remain unchanged; this is a narrow
observed preset contrast, not compaction or a general quality claim.

## Live recovery preparation — 2026-10-21 UTC

The [frozen protocol](evals/projects/duodoro/RECOVERY_PROTOCOL.md) prepares one
source/test-focused Duodoro request variant with unchanged external formatter
acceptance. It reserves two identical half-budgets from Jorge's original total,
observes a real edit before cancellation, then plans actual CLI restart/review/fork
and eligible chained single-file reversal. The combined scripted regression and
pinned preparation gates are mechanics evidence; no live phase has run. Missed
cancellation, blocked/failed forks, unknown interrupted usage and rejected reversal
remain possible outcomes, with no replacement or budget transfer. Original trial
counts/no-go and G2–G5 remain unchanged; this does not approve a release or establish
a new useful coding envelope.

## Live recovery observation — 2026-10-21 UTC

The [one frozen CLI probe](evals/projects/duodoro/RECOVERY_RESULTS.md) observed a
real formatter edit followed by explicit cancellation, API restart, saved reviews,
guarded fork/check/completion and passing external artifact acceptance. The fork
did not alter the already-written patch. Eligible source reversal restored its
before bytes/mode while preserving pre-existing/later unrelated work; another
restart retained identical run/archive/reversal evidence without execution.
The source has an unknown interrupted-request usage tail, so do not certify a known
within-budget token success. Exactly two reserved half-budget phases ran, with no
transfer, replacement or coding rescue. All owned APIs exited and Gemma unloaded.
This supplies narrow live CLI partial-work/reversal evidence missing from the
original cohort; one selected request variant is not broader G2 client/ten-task
acceptance, general recovery/quality, G3/G4 success or G5 approval. Original
trial/comparison outcomes and unattended no-go remain unchanged.

## Panel recovery protocol — 2026-10-21

The [panel protocol](evals/projects/duodoro/PANEL_RECOVERY_PROTOCOL.md) specifies
one browser workflow with the same source/test-focused request and five-minute
total reservation as the CLI probe. The unrun plan is not an execution freeze:
fresh preparation, real-browser scripted preflight, pinned G0 and independently
reviewed final manifest remain required before inference. It names existing SSE,
saved review, continuation and reversal controls plus their storage/timeout limits;
an arbitrary forgotten reversal-key lookup is not available in the panel. No
workflow result, runtime feature, task-quality gain or release approval is claimed.
The original cohort/no-go and G2–G5 remain unchanged.

## Concrete audit findings and scope decisions

| Finding | Classification | Disposition |
|---|---|---|
| Wheel omits the panel's `ui/` assets. `UI_DIR` is relative to the installed API package and mounts only when present. | **Blocks a packaged release advertising the panel.** Does not block the source-checkout trial. | Reproduced below. Do not claim wheel-installed panel support; package assets and test outside the checkout in a focused blocker fix if wheel distribution is selected. No packaging code changed here. |
| `demo.md` still called selectable model profiles future work. | Stale onboarding. | Corrected to startup-selected named profiles; runtime switching remains deferred. |
| Roadmap “Review baseline” table described pre-pivot shortcomings as current, and broad unchecked boxes obscure substantial shipped slices. | Stale/ambiguous status presentation. | Mark that table historical and link this evidence map. Keep broad composite boxes open instead of checking incomplete milestones from test count. |
| Main wall/context/total-token budgets default to disabled; exact project checks and required verification are configured opt-ins. | Operational trial prerequisite, not a claim that default startup enforces chosen release budgets. | Use explicit frozen settings and the configured project check; confirm them in captured execution evidence. No defaults/tuning changed by the audit. |
| Guarded continuation/reversal and scope checks cannot reconcile all external writers, dependencies or arbitrary command side effects. | Known trusted-local boundary. | One process/writer and disposable copies for trial. Cross-process coordination, hostile-code isolation and atomic whole-workspace reversal stay out of this proposed trial, not secretly implemented. |
| No demonstrated context compaction, ten-task client trial or completed held-out comparison. | Open acceptance, not something offline tests can close. | Track G2–G4 and the long-task limitation; do not create a release-success claim. |

## Scripted panel preflight — 2026-10-21 UTC

The [browser preflight](evals/projects/duodoro/PANEL_PREFLIGHT_RESULTS.md) passed
cancel/restart/review/guarded fork, independent final acceptance, two eligible
file reversals and retained-key inspection using a fake provider and real tools.
No inference ran. Forty-six focused regressions passed; this is not a fresh full
G0, live-model quality evidence or closure of G2. Configured `gemma4:12b` identity
in the panel reflects settings, not the actual scripted provider implementation.
The next prerequisite is pinned G0 and an independently reviewed actual live
execution manifest. Original cohorts, release no-go and G2–G5 remain unchanged.

## Live panel execution preparation — 2026-10-21 UTC

The [execution freeze](evals/projects/duodoro/PANEL_EXECUTION_FREEZE.md) now records
fresh actual inputs and first-request guards for one bounded panel workflow.
Pinned G0 passed 53 focused/1,363 offline tests (five live deselected), static
checks and both scripted 90/90 matrices. Scheduled stores remain absent, model
unloaded and no API running; the owned browser tab retains its generated owner.
No inference ran. Independent review/merge and unchanged launch preconditions are
required before the separate execution slice. G2–G5 and release no-go remain open.

## Live panel recovery observation — 2026-10-22 UTC

The [one frozen panel workflow](evals/projects/duodoro/PANEL_RECOVERY_RESULTS.md)
passed native SSE write/cancel, API restart, saved review, guarded fork/check,
independent final acceptance, eligible source reversal and restart lookup without
replay. Original target bytes/mode and later preservation work survived; records
and criteria stayed unchanged. The fork checked an already correct source patch,
not an incorrect-patch repair. Source cancelled usage has an unknown tail; no
known aggregate token-compliance claim. Owned APIs/tab stopped and Gemma unloaded.
Together with the CLI result this is one observed workflow per client, not broad
G2 reliability, ten-task acceptance, held-out quality or release approval. Original
cohorts/no-go and broad G2–G5 remain unchanged. No new application behavior or G0
rerun is claimed by this docs/data report.

## Whole-file envelope follow-up preparation — 2026-10-22 UTC

The [new protocol](evals/projects/duodoro/WHOLE_FILE_TEN_TASK_PROTOCOL.md) keeps the
original ten exact requests/clients/criteria and limits, selecting the existing
whole-file preset to test the broader candidate envelope. All ten fresh initial-
fail/reference-pass pairs and historical request/criterion/reference hashes match;
28 focused criterion regressions passed. No inference or actual execution freeze.
Whole-file client preflight, pinned G0 and independent execution-manifest review
remain required. Original failed counts and G3 budget/source-review requirements
are unchanged; no held-out/causal improvement or release approval follows. This
is the next evidence-gathering slice, not another feature batch.

## Whole-file execution preparation — 2026-10-22 UTC

The [actual freeze](evals/projects/duodoro/WHOLE_FILE_EXECUTION_FREEZE.md) pins the
ten fresh roots, absent scheduled stores, profiles/first-input guards, actual
browser owner, environment/model/assets and recipes at `d1e5778`. Corrected
scripted CLI T03/T05 and panel T04 preflight passed with real tools, separate
stores and retained helper-error evidence. G0 passed 82 focused/1,363 offline tests
(five live deselected), static checks and both scripted 90/90 matrices; forty
initial-input mismatches were rejected without forwarding. No inference ran.
Temporary servers stopped; only the session-free owner tab remains for execution.
Independent review/merge and unchanged launch checks remain. This does not close
G3/G4/G5 or change historical no-go/counts; actual ten-task outcomes are unknown.

## Validation performed for this audit

At the audited source revision, with only documentation being prepared:

- 162 focused existing regressions passed: startup, persistence, runs, cancellation,
  continuation, reversal, check outcomes, loop budgets/recovery, processes,
  inspection and profiles. These use deterministic providers; no real inference.
- 1,256 offline tests passed, five live tests deselected. Ruff, Python formatting,
  mypy (56 core source files), and both JavaScript syntax checks passed. The only
  test warning was the existing Chroma asyncio deprecation. No evaluation behavior
  changed; scripted matrices were not rerun for this documentation-only audit.
- Wheel build: `uv build --wheel --out-dir <temporary-directory>` succeeded. The
  artifact had 179 entries and **zero `ui/` entries**. Extracting it into a temporary
  directory, importing its API outside the checkout with isolated Python path
  precedence, and inspecting the routes reported `UI_DIR.is_dir() == False` and
  no `ui` mount. No API lifespan/provider was started. This is an asset-packaging
  reproduction, not a full clean-environment installation or dependency audit.
- Python 3.14.5/macOS was used locally. Repository CI is configured for Python 3.11
  on Ubuntu; that is distinct from a newly executed full compatibility matrix.
- No new model calls, paid calls, Duodoro changes, Docker actions or release
  publication. Historical browser checks through #143 were inspected, not rerun
  as a fresh real-model/client workflow in this documentation audit.


## Whole-file ten-task follow-up — 2026-10-22 UTC

The reviewed [whole-file cohort](evals/projects/duodoro/WHOLE_FILE_TEN_TASK_RESULTS.md)
ran all ten original tasks once at frozen `d1e5778`: **2/10 accepted artifacts,
2/10 completed-and-accepted, 2/10 acceptable with known reported usage within
nominal budgets**. Seven usage totals are unknown. Five provider read timeouts,
a reserved-fork dispatch stop, missing output usage and unsafe-integer acceptance
failure are retained without repair/replacement. All external grades reproduced;
protected files/modes and stored evidence were preserved. T03 cancelled before
an edit and its fork made no patch. T05 had no eligible reversal difference;
its declared later sentinel survived restart. These controls do not close broader
G2. Fifteen owned APIs stopped, Gemma unloaded and tab closed.

G3 misses the unchanged eight-of-ten target; retain no-go and feature freeze.
G4/G5 remain unmet. Original trial/comparison counts and this point-in-time audit
remain unchanged. A narrower supported envelope or bounded offline investigation
needs an explicit next decision; this cohort does not authorize a release or
establish a causal/held-out gain. Snapshot observation failures and reporting
corrections are documented, with native client submissions/SSE and read-only DOM
observations retained. No application/evaluator behavior changed in this report.
