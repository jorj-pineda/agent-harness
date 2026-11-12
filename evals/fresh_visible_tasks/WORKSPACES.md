# Fresh visible-check workspace selection

The [development proposal](README.md) now has a bounded internal workspace seam
through `evals.fresh_visible_workspaces` and the existing `run_task` evaluator.
It starts no backend, transport, provider request or scheduler and grants no
inference authority. The normal evaluator entry and older prepared policies keep
their contracts. This does not turn an authoring validation into model evidence.

`v1/selection.json` selects the twelve proposed ordinals in balanced paired order,
with the catalog hash and each task's initial fixture, independent acceptance,
reference and stronger-test tree hashes. Tree pins include file modes. Its own
expected byte digest is fixed in code. Selection reobserves all three tasks and
refuses changed requests, criteria, implementation or visible tests before returning
one cell. These are selected local bytes, not authentication or contamination proof.

`prepare_fresh_cell` copies the selected original implementation and replaces only
`test_visible.py` for stronger cells. It reserves a new evaluation-owned directory
outside Git working trees, writes the matching receipt last and refuses occupied
outputs. The prepared container is private; verification checks exact inventory,
receipt, arm-specific bytes and modes. Failure preserves partial files rather than
resetting them. Preparation supplies no runtime/backend image or execution receipt.

`admit_fresh_cell` snapshots that arm-specific starting workspace, writes an
exclusive admission marker before yielding and rechecks selection/contents. Replay
refuses; failures after the marker consume the cell and retain its workspace.
These fixture markers are not actual comparison execution authority or admission.
A future execution layer must enforce actual approval and owned resources before
calling the shared evaluator. Native capture rehearsals may use scripted responses.

The evaluator's explicit `fresh_cell` option is mutually exclusive with existing
prepared/public/feedback options. It binds task identity (including request and
catalog), harness mode, repeat and the existing selected public coding policy.
The guard reobserves the policy source and task/criterion selection before provider
calls and around original independent grading; failure latches even if restored.
Failed post-grade preservation retains the observed grade but prevents acceptance.
Full Settings/host/model/dependency selection remains a future execution-freeze
responsibility; the existing policy subset alone does not freeze those inputs.

The initial visible-test replacement is excluded from agent changes by the
arm-specific baseline. Later test edits are recorded against that selected starting
test, so weakening cannot disappear into setup. External independent acceptance is
unchanged in both arms and remains outside the editable workspace. New invocations
add `fresh_visible_preservation` and `prepared-fresh-visible-single-use-v1` labels,
ordinal, source-selection digest, checkpoint state and retained workspace path.
Existing invocation labels are unchanged.

Scripted real-file regressions cover all twelve preparations/single-use admissions,
reference success across all tasks/arms and both repeats, weak-pass/stronger-fail
partial patches, visible-test weakening with independent rejection, pre-admission
drift and invalid policies, mid-run source drift, latched failure and post-grading
policy drift. Prior pagination-workspace regressions also run. Canned responses
include explicit token usage; unknown usage remains subject to existing preventive
runtime guards. This validates mechanics, not model quality or native settlement.

Next is mandatory-mock input capture and owned-lifecycle rehearsal for this exact
selection, then a retained clean candidate/full gate, concrete freeze and actual
approval/fresh launch checks. There is no new inference authorization, paid call,
retry, model default, process isolation, held-out claim or release gate closure.
