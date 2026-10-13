# Duodoro real-project validation

Jorge authorized the local Duodoro repository. Pin the inspected source at
`5324cabba4ad3c8ad2e1cd1480218cf087d9b741`; do not edit its working checkout,
reset user changes, deploy, start account/database services, or touch Docker.

The first frozen task is **duration formatting** (the default): negative/non-finite numeric
values render as zero, while finite-positive duration/countdown formatting and
`formatTag` stay compatible. Only `client/src/lib/format.ts` may change. The full
pinned source is copied, with shared disposable-workspace exclusions; a visible
Python/Node check and representative pre-existing-work file are added before the
baseline. Existing source/tests/scripts and those setup files must remain unchanged.

Independent acceptance stays outside the model's workspace. It checks invalid
inputs, 18 positive/fractional/boundary values, tag compatibility, exact file scope
and preservation of baseline contents/modes. Criterion hashes are frozen before
inference and must be checked afterward. A reference patch is also outside the
workspace. Baseline and reference must respectively fail/pass both visible and
independent checks before any inference. These are trusted host processes, not
process isolation; hashes/file-tool boundaries do not make arbitrary repository
code unable to access the host.

The adapter uses installed Node's native TypeScript support (validated locally on
managed Node 24.21.0). It runs library checks without installing dependencies or running
Next/Vitest, authentication, browser, database or server integration suites. Duodoro
normally uses Node 22 in CI. This task does not establish full application regression
coverage or runtime compatibility on other Node versions.

Preparation reuses `evals.real_run.Task`, `validate_task` and the existing shared
runtime/real-task runner. No provider calls occur during preparation:

```sh
uv run python - <<'PY'
import asyncio
import tempfile
from pathlib import Path
from evals.projects.duodoro.prepare import prepare_task, acceptance_unchanged
from evals.real_run import validate_task
parent = Path(tempfile.mkdtemp(prefix="duodoro-task-"))
task = prepare_task(Path("/Users/jorge/Projects/duodoro"), parent / "prepared")
asyncio.run(validate_task(task))
assert acceptance_unchanged(parent / "prepared")
print(parent / "prepared")
PY
```

The caller owns cleanup of the prepared directory. Keep prepared artifacts and
reports at unique paths outside the source checkout. Before a model run, record the
harness revision, pinned task/criterion hashes, exact provider/model/sampling/context
and tool/token/time budgets. Use `run_task` with the same prepared task/settings in
both modes. Require unchanged criterion hashes and reject added/deleted files or
modified paths other than the permitted target, even if executable assertions pass.
Include every attempt and distinguish environment/setup failures from patch failures.
Retain raw traces/diffs and record manual rescues. Jorge authorized local `gemma4:12b` and a cheaper model through his existing
OpenCode account within account limits. Use a currently free Go model for the next
bounded comparison; do not silently switch to paid usage or another repository.
Stop any evaluation-owned Ollama processes afterward, preserving user-owned services.

The second frozen task is **timer normalization**. Select it with
`prepare_task(repository, destination, task_name="timer_normalization")`.
Only `server/payloadParsers.js` may change: floor final defaulted/clamped durations,
preserving coercion, falsy/NaN defaults, Infinity clamps, mode/session-ID handling
and other exported parsers. External acceptance checks 26 duration inputs across
six mode values, invalid session IDs, Symbol error behavior, representative other
parser contracts and exact file/content/mode preservation. The visible adapter
covers fractional normalization and positive Infinity; it is intentionally narrower.
Initial/reference fail/pass both checks. This uses dependency-free CommonJS/Node
library checks, not Vitest, Socket.IO, database or full server integration.

The third frozen task is **calendar layout**. Select
`task_name="calendar_layout"` to add an optional boolean `sundayFirst` argument to
`monthDays`, defaulting false. Only `client/src/lib/focusCalendar.ts` may change;
UI wiring is outside the task. External acceptance independently generates 168
complete grids with Python's standard calendar library across seven representative
years (including leap/century boundaries), checks default/explicit Monday layouts,
other calendar exports and exact file/content/mode scope.

The native TypeScript adapter uses a narrowly scoped synchronous Node resolution
hook for the existing extensionless `./focusTags` import from `focusCalendar.ts`.
It loads real library modules, without rewriting source or simulating function
results. This hook was validated with the managed command environment; the shell
Node version can differ. See the [Node customization-hook documentation](https://nodejs.org/download/release/v24.0.1/docs/api/module.html).
Both initial checks fail and the reference passes. Native runtime checks omit
Next/Vitest, full integration and TypeScript typechecking; Node 22 compatibility
is unverified. No model inference on this task has run yet.

These are three prepared real-project tasks, not completion of the ten-task release target,
held-out evaluation or repeated comparisons on multiple lower-tier models. Offline
baseline/reference validation is criterion validation, not model-quality evidence.
