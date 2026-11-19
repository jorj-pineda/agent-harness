# Current release checklist

Snapshot reconciled on 2026-11-19, including the merged merge-discipline change,
scripted CLI process evidence and the retained, unapproved fresh comparison.
This is the short operational view of the [roadmap](roadmap.md), which still
governs scope and acceptance criteria. The [release audit](RELEASE_READINESS.md)
retains historical findings and limitations. This checklist grants no execution
or publication approval.

**Disposition: experimental, trusted-local, supervised use. The agreed unattended
coding release remains a no-go.** There is no evidence-based release date or
completion percentage. Infrastructure exists; useful coding reliability and full
workflow acceptance remain incomplete.

## Release gates

| Gate | Current state | Evidence needed to close it |
|---|---|---|
| G0: candidate mechanics | Historical selected runtimes pass; must be revalidated for each candidate. | Full offline tests, Ruff, formatting, mypy, both scripted matrices and relevant installation checks on the exact selected candidate. Keep scripted results labeled as mechanics. |
| G1: agreed scope | Original ten-task agreement complete; the separate six-cell public pilot is closed. | Freeze any new task/model/settings/budget selection and independent criteria before inference. Prior approval does not authorize new cells or retries. |
| G2: client and recovery workflow | Partial implementation and bounded live observations; broader acceptance open. | Demonstrate agreed CLI/panel submission, review, cancellation, restart, guarded continuation and reversal cases, recording interventions and preservation of pre-existing work. |
| G3: useful coding envelope | Unmet. Latest complete original-project follow-up accepted 2/10 against 8/10 required. | At least 8/10 agreed bounded real tasks accepted without manual patch repair, within explicitly chosen budgets. A narrower scope needs explicit agreement and must retain the original unmet target in reporting. |
| G4: repeatable quality | Unmet. Public selected-task observations do not establish held-out improvement. | Repeated same-model baseline comparisons under comparable budgets, fresh held-out coverage, intended-model/category results and the roadmap's at-least-two-model measurements. Publish raw counts and variation. |
| G5: distribution and release decision | Packaged assets/fresh installs have historical evidence; release acceptance open. | Reconcile all open gates; validate the chosen distribution and onboarding on the candidate, document supported scope/limitations, and obtain Jorge's release decision. |

The roadmap also requires zero false verified states in its adversarial regression
suite and cancellation/restart/revert checks with no loss of pre-existing work.
These are scoped mechanics/workflow claims, not universal semantic correctness.
Passing a configured project check does not prove independent task acceptance.

## Latest observations

- The [visible-check comparison](evals/public_feedback/RESULTS_20261112.md) records
  control 0/4 and bool_check 4/4 accepted/completed-and-accepted on one known task
  and one model. All copied grades match and eight settlements are clean/unforced.
  Treatment fixes bool before its first check; no corrective-feedback recovery is
  demonstrated. The limited rule is eligible for human review, benefit remains
  `not_assessed`, and this closes no release gate or original unmet target.

- The [original-project timeout follow-up](evals/projects/duodoro/PROVIDER_WAIT_TEN_TASK_RESULTS.md)
  accepted and completed-and-accepted 2/10 artifacts. Its original criteria and
  budgets remain authoritative for that cohort.
- The [public pilot](evals/public_envelope/RESULTS_20261109.md) accepted and
  completed-and-accepted 4/6 artifacts: divide-zero 2/2, label refactor 2/2,
  pagination 0/2. Six fresh copied-workspace regrades match. This is one policy,
  three known tasks and one model; it does not supersede the ten-task results.
- Both pagination attempts miss the explicit bool restriction, pass the visible
  check, write again and end at the iteration cap with stale verification.
  The runtime reports the unfinished state; independent grading rejects the patch.

All these cohorts are closed. Unrun/stopped cells in older cohorts are not retry
slots. Recorded preparation, scripted rehearsal and live coding results remain
separate kinds of evidence.

## Completed preparation and current boundaries

- The [closed visible-check comparison](evals/public_feedback/RESULTS_20261112.md)
  supersedes its earlier preparation queue. Its control 0/4 versus bool_check 4/4
  result remains bounded known-task evidence, with benefit `not_assessed`.
- The [fresh retained candidate](evals/fresh_visible_tasks/RETAINED_CANDIDATE_20261116.md),
  [merged freeze/operator](evals/fresh_visible_tasks/FREEZE_20261116.md), and
  [scripted review exercise](evals/fresh_visible_tasks/REVIEW_EVIDENCE.md) complete
  preparation for twelve selected cells. The retained candidate's 3,840-test G0 is
  historical evidence for its selected runtime, not today's checkout. All twelve
  scripted reviews are unknown/incomplete; no model-quality benefit is established.
  Jorge selected offline work. Actual approval and fresh launch/resource checks
  remain required before generation; this page grants neither.
- Real CLI/API process tests now cover restart review, SIGINT, shutdown, guarded
  continuation/reversal, disconnect, timeout recovery, command descendants,
  contention and crash recovery. The [roadmap](roadmap.md) records their individual
  limits. These use a scripted provider and do not close broader G2 or G3–G5.
- [Merge discipline](.github/MERGE_DISCIPLINE.md) is implemented in merged PR #450
  (`f417a3e`). Main protection was applied and read back on 2026-11-18. Both required
  Actions checks passed on PR head `01d9896`; downloaded coding/support reports
  identify their respective sources and report contract PASS. Local validation
  passed 4,027 offline tests (five live deselected), static checks and both scripted
  matrices. These results are revision-specific mechanics evidence, not a fresh
  G0 for the retained candidate or a release decision. No deployment was added.

## Immediate queue

1. The [G2 evidence map](WORKFLOW_ACCEPTANCE.md) separates existing scripted,
   process and historical actual-model observations. Permission-only stale-state
   refusal after restart is the next bounded offline CLI acceptance case; broader
   candidate-specific CLI/panel workflow acceptance remains open.
2. Exercise the highest-priority remaining offline workflow case with the real
   application and disposable user state, preserving bytes, modes and Git state.
   Fix runtime behavior only if the probe demonstrates a defect. Passing a new
   scripted case closes that case, not the entire workflow or coding-quality gate.
3. Keep the fresh twelve-cell comparison unadmitted while offline work is selected.
   Any future generation needs actual approval of the exact freeze and fresh
   launch checks. Even a successful fresh development-task comparison would not
   replace the original 8/10 target, held-out or second-model measurements.
4. Revisit distribution/onboarding validation after the workflow and coding-quality
   evidence supports a release decision. No automated deployment is required for
   this offline batch.

Do not add a premium supervisor, agents, semantic search or a larger prompt merely
to fill the queue. Do not promise an unattended release from public pilot success.

## How to maintain this page

Update after an accepted result or release decision, link the source evidence,
and state what changed in a gate. Keep historical runtime/test counts attached to
their original revisions. Do not close gates based on PR counts, mocked success,
passing editable tests or a model's completion prose. Keep the detailed roadmap
history instead of rewriting closed results.
