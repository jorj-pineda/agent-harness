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
they do not accept these prepared copies yet. Connecting a prepared root to input
capture/execution requires fresh verification and explicit evaluation-owned
admission in the next slice. A receipt alone is never permission to edit an
unchecked workspace, reuse a completed attempt or launch inference. The actual
reviewed execution freeze and approval gates remain incomplete.
