# Current release checklist

Snapshot after [PR #391](https://github.com/jorj-pineda/agent-harness/pull/391),
including the [offline pagination diagnosis](evals/public_envelope/PAGINATION_DIAGNOSIS_20261110.md).
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

## Immediate queue

1. The retained pagination diagnosis is complete: the request includes the bool
   restriction, the visible check omits it, and both artifacts admit True. Exact
   duplicate writes explain stale verification but do not cause that rejection.
   Do not change verification rules or budgets to relabel these failures.
2. The [public visible-check comparison proposal](evals/public_feedback/PROTOCOL.md)
   selects eight fresh cells across original and bool-covering visible checks, with
   unchanged task/acceptance and per-cell budgets. Exact fixture validation, fresh
   shared-evaluator workspaces and [mandatory-mock capture/rehearsal](evals/public_feedback/REHEARSAL.md)
   are implemented. The recorded eight scripted reference successes prove mechanics,
   not feedback benefit. [Capture verification/preservation and exact-path staging](evals/public_feedback/STAGING.md)
   and [execution selection/preflight/admission](evals/public_feedback/EXECUTION.md)
   are implemented without launching inference or authenticating approval assertions.
   [Admitted provider guards](evals/public_feedback/PROVIDER_GUARDS.md), the fixed
   schedule and [owned serial execution](evals/public_feedback/OWNED_EXECUTION.md)
   are implemented. A native mandatory-mock owned rehearsal settled eight cells
   cleanly; its scripted successes establish mechanics, not model quality.
   Next is fresh candidate preparation/rehearsal/probes, concrete reviewed freeze,
   CI/merge, actual approval and fresh launch gates.
   This selects no application default and authorizes no inference.
3. If that scope demonstrates repeatable usefulness, validate held-out/model
   coverage and the real client/recovery workflow, then validate the distribution.

Do not add a premium supervisor, agents, semantic search or a larger prompt merely
to fill the queue. Do not promise an unattended release from public pilot success.

## How to maintain this page

Update after an accepted result or release decision, link the source evidence,
and state what changed in a gate. Keep historical runtime/test counts attached to
their original revisions. Do not close gates based on PR counts, mocked success,
passing editable tests or a model's completion prose. Keep the detailed roadmap
history instead of rewriting closed results.
