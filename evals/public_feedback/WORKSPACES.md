# Fresh comparison workspaces and evaluator binding

`evals.public_feedback_workspaces.prepare_feedback_cell(ordinal, directory)`
selects one of the eight fixed [plan](plan.json) cells and copies the original
pagination fixture to a fresh named directory outside Git workspaces/source
inputs. Control retains the original visible test. Treatment replaces only
`test_visible.py` with the [selected bytes](VALIDATION.md), retaining the original
implementation, request, reference and independent acceptance.

The receipt binds the comparison plan, ordinal/arm/repeat and expected initial
tree digest. Expected file hashes/modes derive from the original fixture plus
the selected visible test. Preparation checks source selection again after copying
and writes the receipt last. Existing destinations refuse; failures retain partial
output instead of resetting or deleting it. Containers use mode 0700 and receipts
0600. These are bounded non-atomic observations, not process isolation.

`verify_feedback_cell` checks the fresh container inventory, receipt bytes/mode
and complete initial file tree. `admit_feedback_cell` captures the change baseline
and writes an exclusive single-use marker before yielding the workspace. A failure
after that marker consumes admission and preserves partial evidence. Workspaces
remain on exit. These mechanics do not authenticate approval or grant a resource
lease, and there is no live comparison CLI.

## Shared evaluator

The existing `evals.real_run.run_task` accepts an explicit `feedback_cell` selection.
It is mutually exclusive with old prepared-held-out and public-pilot policies.
Default disposable evaluation and existing public-pilot behavior remain unchanged.
Before admission, the guard checks the original task identity, harness mode,
selected repeat and fixed runtime policy. Ordinal identifies the arm through the
pinned plan; a caller cannot substitute a different task/request or visible test.

The selected workspace uses the application runtime, registry and configured
project check. The original external grader still controls acceptance. Source and
policy checks run before model requests and before/after grading; failure latches
and prevents accepted status, even when an available grading result passed. The
guard binds the selected policy subset, not a complete host/Settings freeze;
complete settings/environment/model/resource/input pinning remains future work.

Runtime metadata records `prepared-public-feedback-single-use-v1`, the actual named
workspace root and `public_feedback_preservation` evidence with the plan/ordinal
and checkpoints. A rejected preservation result retains the grader's output when
available. This is evaluation provenance, not a correctness probability.

The initial selected visible test belongs to the baseline. It is not counted as
an agent edit merely because the treatment differs from the original fixture.
For later agent edits to that test, diffs use its bounded captured initial contents;
other paths keep their original task baseline. Weakening the visible test may
produce a visible pass but cannot redefine independent acceptance.

## Validation and limits

Scripted-provider tests use real copied files, atomic writes, configured pytest
checks and original independent grading. They cover all eight fresh cell selections,
repeat/arm binding, single-use/occupied-path refusal, modified initial files/receipt/
inventory/modes, invalid ordinals and Git-path refusal. A simulated post-marker
failure retains the marker and source copy. Task/mode/repeat/policy conflicts refuse
before requests or admission; midrun and after-grading policy drift latch closed.

Reference workflows in both arms pass original acceptance. A synthetic reproduction
of the observed bool defect is rejected in both arms, despite control's visible
pass. A treatment-test weakening case records the correct diff and is independently
rejected. These prove harness/evaluator mechanics, not model recovery or a quality
gain. No real inference is part of this slice.

Named workspaces and path checks do not isolate executed code from the host.
Only trusted selected fixtures belong here. No old original artifacts, acceptance
criteria, frozen plans, application defaults or closed cohorts change.

Mandatory-mock rehearsal and normalized/native first-input capture are the next
preparation slice. Owned execution, complete retained preparation, concrete reviewed
execution freeze, actual approval and fresh launch gates still precede inference.
The original release targets/no-go and G2–G5 remain unchanged.
