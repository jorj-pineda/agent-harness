# Offline comparison selection and visible test

`evals.public_feedback.validate_feedback_plan()` checks the exact reviewed plan
bytes, strict schema and original public plan/task/catalog/budget selection. It
reuses the original public validator, including its bounded tree hashes/modes;
unrelated original public task drift also refuses rather than weakening that
existing selection. No old inputs are rewritten.

The separately selected [fixture pin](fixture.json) binds the exact treatment test
bytes and mode. The [test](fixtures/test_visible.py) retains the two original
visible assertions and adds eight parametrized bool cases from the explicit
public task request. It changes neither the starting implementation nor the
independent acceptance criteria. The protocol's semantic plan remains unchanged.

`visible_check_bytes("control")` returns the original test unchanged;
`visible_check_bytes("bool_check")` returns the selected treatment bytes. Unknown
arms refuse. Validation brackets extraction; the bounded regular-file reader
refuses drift, oversize and final-component symlinks. These are bounded non-atomic
file observations, not an external-writer lock or OS isolation.

These functions are offline selection only: they create no workspace, provider,
backend, output artifact or admission. Fresh arm-specific workspaces, shared
evaluator binding and mandatory-mock capture/rehearsal are still to be implemented.
No application settings/defaults, original fixture/catalog/acceptance, closed
cohort or live execution authority changes.

## Evidence for the test's discrimination

The focused regression suite uses a synthetic implementation of the observed
`isinstance` bool defect and the unchanged known reference. Control passes both;
the treatment rejects the defect (4 failed, 6 passed) and accepts the reference
(10 passed). Drift regressions exercise plan, treatment bytes/modes/symlink/size,
fixture selection and original implementation/acceptance/reference trees.

Separately in this implementation session, copied **actual retained artifacts**
from closed cells 2 and 5 passed control and failed the selected treatment with
4 failed/6 passed each. Unchanged original independent acceptance still rejected
both copies. The unchanged reference passed all 10 treatment cases and all 21
original acceptance cases. All 159 selected original evidence hashes matched
before and after these probes; no original artifact was repaired or resubmitted.
The local proof recipe/output are `/tmp/prove-feedback-validation.py` and
`/tmp/agent-harness-feedback-validation-proof.json`.

These real-file checks establish that the selected visible test can expose the
observed defect. They do not show a model responding to it or any coding-quality
gain. The synthetic regression and actual copied-artifact probes are distinct
from a live-model comparison; there were no model requests.

Before inference, this comparison still needs fresh shared-runtime workspaces,
complete settings/native first-input/resource capture, mandatory-mock rehearsal,
owned execution, concrete retained preparation/freeze review, actual approval and
fresh launch gates. Release no-go and G2–G5 remain unchanged.
