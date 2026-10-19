# Release-readiness audit — 2026-10-18

Audited source: `62ac6a44f1a6d96e7ea261162d51bc662306cefa` (merged through PR #143).
This is a point-in-time source, regression and recorded-evidence audit. It is not
an independent security audit, a fresh model comparison or release approval.

**Decision: suitable for a controlled, supervised trial from a source checkout;
not yet ready to claim the roadmap's personal-use release targets.** Infrastructure
is substantially implemented. Reliable coding performance and the complete real
workflow are still unproven. PR/test counts are not a release-completion percentage.

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

| Gate | Current disposition | Evidence required to close it |
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

## Concrete audit findings and scope decisions

| Finding | Classification | Disposition |
|---|---|---|
| Wheel omits the panel's `ui/` assets. `UI_DIR` is relative to the installed API package and mounts only when present. | **Blocks a packaged release advertising the panel.** Does not block the source-checkout trial. | Reproduced below. Do not claim wheel-installed panel support; package assets and test outside the checkout in a focused blocker fix if wheel distribution is selected. No packaging code changed here. |
| `demo.md` still called selectable model profiles future work. | Stale onboarding. | Corrected to startup-selected named profiles; runtime switching remains deferred. |
| Roadmap “Review baseline” table described pre-pivot shortcomings as current, and broad unchecked boxes obscure substantial shipped slices. | Stale/ambiguous status presentation. | Mark that table historical and link this evidence map. Keep broad composite boxes open instead of checking incomplete milestones from test count. |
| Main wall/context/total-token budgets default to disabled; exact project checks and required verification are configured opt-ins. | Operational trial prerequisite, not a claim that default startup enforces chosen release budgets. | Use explicit frozen settings and the configured project check; confirm them in captured execution evidence. No defaults/tuning changed by the audit. |
| Guarded continuation/reversal and scope checks cannot reconcile all external writers, dependencies or arbitrary command side effects. | Known trusted-local boundary. | One process/writer and disposable copies for trial. Cross-process coordination, hostile-code isolation and atomic whole-workspace reversal stay out of this proposed trial, not secretly implemented. |
| No demonstrated context compaction, ten-task client trial or completed held-out comparison. | Open acceptance, not something offline tests can close. | Track G2–G4 and the long-task limitation; do not create a release-success claim. |

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
