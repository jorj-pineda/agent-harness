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


## Completed frozen client trial

The [2026-10-19 results](TRIAL_RESULTS.md) retain the full real-client local Gemma
cohort: **2/10 accepted artifacts, 1/10 completed-and-accepted**, below the 8/10
target. G1 review completed before inference; G2 remains open (real reversal was
not demonstrated), G3 was missed and G4/G5 remain unmet. No Duodoro working-tree
edits or paid calls occurred. The configuration is a no-go for the frozen
unattended task envelope. This is a public development cohort, not held-out or
comparative quality evidence.

### Preparation record

The [ten-task protocol](TRIAL_PROTOCOL.md) and [unrun ledger template](trial-ledger.json)
record Jorge's approval of five CLI and five panel tasks, local Gemma, the
five-minute dispatch allowance and the proposed token/recovery limits. All ten
adapters validate initial failure and reference success offline. The seven new
factory task names are `share_storage`, `timer_prefs`, `shortcut_count`,
`rotation_preview`, `duration_refactor`, `tag_filter` and `wrapup_intentions`.
Original three task definitions and acceptance criteria remain unchanged.

Checks use native Node 24 (preparation validated on 24.22.0) for library behavior;
they do not establish full-app, Vitest or TypeScript typecheck coverage. Portable
pinned source snapshots support offline CI without the user's Duodoro checkout.
External criteria reject selected incorrect patches that pass visible smoke tests.
T08 additionally needs independent source review using its external rubric: an
automated passing behavioral grade alone is insufficient for full acceptance.

The [deterministic client preflight](CLIENT_PREFLIGHT.md) passed representative
CLI/panel workflows without inference. The [execution freeze](EXECUTION_FREEZE.md)
records fresh baselines, exact requests/settings/model metadata and G0 at runtime
`cd98917`; review completed in PR #148 before authorized execution. Earlier free-Go authorization is not a current pricing
claim or instruction to start hosted calls. No model ran during this preparation;
these public development tasks are not held out and establish no quality gain.

Preparation validation: 41 focused tests and 1,291 offline tests passed (five live
tests deselected); Ruff, formatting and mypy passed. Both scripted matrices passed
90/90 escalation decisions at unique temporary report paths; their reports retain
scripted-contract labels and metric limits. No inference or model comparison ran.


## Completed paired feedback follow-up

The reviewed [comparison freeze](EDIT_FEEDBACK_COMPARISON.md) prepared twelve attempts on
T01/T04/T09 with unchanged requests, clients, acceptance and per-task limits.
Two repeats per arm counterbalance pair order. Control and treatment share the
merged provider-failure reporting runtime; the only runtime difference is the
old/new mismatch diagnostic. Both-arm offline checks and the declared five-case
control exclusion are recorded. The [2026-10-20 results](EDIT_FEEDBACK_RESULTS.md)
retain all twelve outcomes: control **0/6 accepted, 0/6 completed-and-accepted**;
treatment **6/6 accepted, 4/6 completed-and-accepted**. All final grades reproduced
and protected files remained unchanged. The rendered workspace paths differ,
including arm names; prefix divergence and one pre-edit control block further
limit attribution. This selected development comparison is not held out and does
not establish a general causal gain. The original trial counts and release gates
remain unchanged. No Duodoro working-checkout edits or paid calls occurred; the
owned API stopped and evaluation-loaded Gemma was unloaded.

The [duration budget diagnosis](DURATION_BUDGET_DIAGNOSIS.md) replays the two
accepted-but-incomplete treatment artifacts offline. Both stops reproduce exactly:
11,057 total tokens remained, while the next prompt heuristic plus minimum output
reservation required 13,018. The root instruction result dominates message bytes,
but no per-request usage was saved, so byte shares cannot attribute provider-token
cost or establish that shortening context would complete the task. No runtime,
prompt, limits or historical outcome changed; per-request client evidence and a
reviewed context experiment remain separate next work.

## Prepared duration read-batching experiment (not run)

The [new freeze](BATCH_READ_COMPARISON.md) prepares four CLI attempts, two per arm,
with the original T01 request/criteria/limits and a single common workspace path.
A local treatment patch adds one sentence encouraging independent read batches;
all instruction and tool-result text stays intact. The production prompt is
unchanged. New per-request budget evidence is available in both pinned runtimes.
An identical evaluation-only provider-boundary assertion rejects unexpected first
inputs before forwarding. Scripted shared-runtime/API probes validate the intended
sentence-only difference, full instructions, real reads/captures and keyed replay.
Both arms pass 1,342 offline tests, static checks and both 90/90 scripted matrices.
Review and a subsequent execution instruction are required before inference.
This is preparation, not a coding result, context-compaction gain or release claim.


## Whole-file ten-task follow-up complete — 2026-10-22 UTC

The [completed cohort](WHOLE_FILE_TEN_TASK_RESULTS.md) retains all ten original
requests/clients/criteria under the reviewed freeze: **2/10 accepted artifacts,
2/10 completed-and-accepted and 2/10 acceptable with known reported usage within
nominal budgets**. All artifact grades reproduced; no patch repair or replacement.
The unchanged eight-of-ten target is missed again. Keep no-go and feature freeze;
broader recovery, held-out/two-model and release gates remain open. Original
cohort counts stay separate. Raw evidence, unknown usage, failed client outcomes,
observation limits and cleanup are recorded in the report.


The [three offline failure investigations](WHOLE_FILE_FAILURE_DIAGNOSIS.md) now
bound the five read-timeout observations, reproduce T05's missing-usage stop and
T06's unsafe-integer defect without inference or artifact repair. Original evidence,
criteria, counts and no-go remain unchanged. Missing backend/wire evidence limits
root-cause attribution; no larger timeout/budget or new trial follows automatically.

## Provider diagnostic source launcher

The evaluation-only `provider_probe` module corrects the startup blocker recorded
in [the failed preparation](PROVIDER_DIAGNOSTIC_RESULTS.md). It loads augmented
rows from `provider-diagnostic-freeze.json`, never raw `tasks.json`, and validates
identity, paths, settings hash and selected configuration before resource allocation.
Both source and review use this loader and the same application component factory;
review rejects inference and source guards the first normalized request unchanged.
The factory's injected component-builder seam supports offline preflight with fake
providers and real API lifecycles; it does not change production harness behavior.

Only after a fresh execution freeze is independently reviewed and merged:

```sh
uv run python -m evals.projects.duodoro.provider_probe \
  --root /absolute/path/to/fresh-reviewed-probe --task T05 --phase source
```

This trusted-operator launcher is not a general evaluation framework, an isolation
boundary, a complete pre-dispatch audit or permission to rerun the old freeze.
Retain the separate hash/model/environment/one-turn launch checks, actual CLI
preflight and unchanged acceptance/budgets. Fresh preparation and live evidence
remain required; no inference or model-quality improvement is established here.

Correction validation: 18 new source-factory/guard regressions passed, with 32
focused including existing API/CLI observation tests. The full gate passed 1,415
offline tests (five live deselected), Ruff, formatting and mypy. Both scripted
matrices passed 90/90 with contract labels/metric limits inspected. The existing
Chroma deprecation warning remains. A read-only loader probe of the historical
blocked preparation found both manifest identities without resource allocation;
old recipes/stores were not changed or executed. These are mechanics checks.


## Transport hypothesis preparation

The [evaluation-only transport protocol](TRANSPORT_PROBE_PROTOCOL.md) proposes a
bounded buffered/streamed comparison after the observed read timeouts. The
experimental wrapper is not selected by application settings and does not enable
a production streaming option. It buffers complete output before tools, records
numeric timing only and retains unchanged task limits. Synthetic HTTP/API tests
are mechanics evidence; fresh same-input/client preparation and an independently
reviewed execution freeze remain prerequisites for inference. Original no-go and
all cohorts remain unchanged.

## Post-plan input diagnostic: preparation only

The [bounded input protocol](REQUEST_INPUT_PROTOCOL.md) follows the completed
zero-chunk transport comparison. Evaluation-only wrappers retain normalized
inputs and exact wire JSON before forwarding, including a later failed request,
with fresh private storage and fixed count/byte bounds. Raw model-visible content
is plaintext and remains local; this is not redaction, backend receipt evidence,
a production setting or a replay permission. The proposed next slice is one fresh
original T09 buffered CLI attempt at unchanged limits after actual-client preflight,
pinned G0 and independent execution-freeze review. No inference ran during this
implementation; the old cohort, no-go and G2–G5 remain unchanged.
