# Named held-out fixture preparation

The input capture revealed that random disposable paths appear in the model's
system message. This preparation utility creates an explicitly named copy without
changing runtime prompts or masking raw input bytes:

```sh
uv run python -m evals.heldout_workspaces --task ho_iso_week --directory /tmp/unique-cell-01
uv run python -m evals.heldout_workspaces --task ho_iso_week --directory /tmp/unique-cell-01 --verify-only
```

Choose a fresh child of an existing directory, outside any Git workspace. The
utility refuses existing destinations, including empty directories and links;
it never replaces, resets or deletes them. Preparation checks the registered
catalog before copying, then verifies the copied fixture against the original
reviewed byte/mode pins. The mode-0700 container holds `workspace/` and a mode-0600
`prepared.json` receipt binding task ID and registered seal digest. The receipt is
written only after successful copy verification. Failed copies remain for inspection.

Verification rechecks the original catalog, receipt/task/seal, bounded container
inventory and copied file bytes/modes. Receipt edits cannot redefine the registered
source pins. Unknown files, special files and symlinks refuse. Canonical parent
resolution makes the actual named location explicit. These are non-atomic path and
file observations; they are not process isolation or external-writer coordination.
The check does not prove historical authorship or absence of task exposure.

No application turn, provider, command or model request is executed. Existing
`run_task` and the generic evaluator still create fresh random disposable copies;
prepared copies require the explicit single-use admission described below. Connecting a prepared root to input
capture/execution requires fresh verification and explicit evaluation-owned
admission. A receipt alone is never permission to edit an
unchecked workspace, reuse a completed attempt or launch inference. The actual
reviewed execution freeze and approval gates remain incomplete.

## Single-use evaluation admission

`run_task(..., prepared_directory=...)` is an evaluation-only seam for registered
held-out tasks. It rechecks task/source/copy identity before creating an exclusive
mode-0600 `admitted.json` marker, then verifies again before exposing the named
workspace to the shared runtime. The original source remains outside the editable
workspace. Changes and baseline identity use the existing workspace machinery;
result configuration records the explicit workspace policy and canonical root.
The default evaluator/application behavior remains fresh disposable copies.

Admission is single-use. Success, failure, cancellation and mock capture all leave
the marker and workspace intact. Subsequent verification/admission refuses rather
than resetting bytes or reusing an attempt. A marker proves only admission was
claimed, not that inference started, completed or settled. It is not crash-safe
or a lock against external writers. Provider ownership stays with the caller.

`uv run python -m evals.heldout_inputs --directory /tmp/new-unique-capture --prepared-workspaces`
prepares and admits a fresh named container per cell, retaining each root beside
its exact normalized/wire input files. The mandatory mock transport still forwards
zero requests. Repeats use distinct cell paths and remain raw-byte unequal; these
are now explicit retained locations, not normalized equivalents. Capture consumes
each container. No reuse, marker deletion, reset or launch permission is provided.
A later operator recipe must preserve consumed evidence and establish a freshly
verified exact-path execution setup under explicit freeze/approval gates.
