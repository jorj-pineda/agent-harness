# Provider diagnostic probe: setup blocker

The reviewed [freeze](PROVIDER_DIAGNOSTIC_FREEZE.md) was merged in PR #180 before
launch. **No model inference or coding submission occurred.** T05 failed API
startup; T09 was blocked by the same reproduced recipe defect. These are setup
outcomes, not two failed coding artifacts or a new model-quality result.

[Structured audit](../../results/2026-10-22-duodoro-provider-diagnostic-blocked.json)
pins the failed recipe/logs and read-only audit. Execution runtime remained
`594ac447188b710eddb07cc930f6c6a496bc9c85`; reporting commits were not executed.

| Scheduled task | Outcome | Coding submission / inference | Acceptance / completion / usage |
|---|---|---|---|
| T05 timer preferences | API startup failed before readiness | None | Unobserved |
| T09 tag filter | Not launched; same setup defect reproduced offline | None | Unobserved |

## Reproduced cause

The frozen evaluation-only `server.py` loads task rows from local `tasks.json`.
That preparation file precedes first-input capture and lacks
`source_initial_identity`. Capture wrote augmented rows to `captured-tasks.json`
and the reviewed execution manifest. The source-mode component factory instead
indexes the unaugmented row and raises `KeyError: 'source_initial_identity'`.

The retained T05 log reports application startup failure and process exit 3.
There is no source CLI invocation, printed submission key or HTTP request log.
The session store was never created; the allocated memory store has zero notes.
Both scheduled workspaces still match their complete initial eligible manifests.
T09's stores/outcomes remain absent. No artifact was graded or modified.

The scripted factory path does not read that field, so the real CLI/tool/restart
preflight passed without exercising the live factory's manifest selection. The
guard unit probes used augmented captured rows directly. Those passing checks
were narrower than the launch path; they did not prove it was ready. This was a
preparation error missed before launch, not an Ollama/harness production failure.
No HTTP duration, native usage or terminal metadata was observed from Gemma.

## Preservation and cleanup

The failed source API PID exited and loopback port 18757 is released. Ollama's
loaded-model list remains empty, so no evaluation model needed unloading; the
existing service was preserved. No retry, replacement, changed recipe, new turn,
budget expansion or task substitution was performed. T09 was blocked without
starting an API known to fail on the same unaugmented rows.

The read-only audit verifies all frozen recipe/input/preflight hashes, exact
requests/settings/criteria/references and both complete workspace manifests.
Duodoro HEAD, status, refs and index fingerprints remain unchanged. Local logs,
failed orchestration recipe and audit are retained at:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-provider-probe-tuxlidi3`

## Next required correction

A separate preparation correction must select and validate the reviewed manifest
row before allocating components, then preflight that **same source factory path**
with a fake/no-forward provider. Include missing identity and raw-versus-augmented
row regressions, cleanup and zero dispatch. Prepare fresh roots/stores and a new
independently reviewed freeze before any future live attempt. Do not repair this
historical freeze in place or reinterpret these setup outcomes as inference.

The audit recipe passed its preservation/zero-dispatch/cleanup assertions. This
docs/data report does not rerun preparation's 60 focused, 1,397 offline/static or
two 90/90 scripted gates; those remain historical mechanics evidence. Original
cohorts, unattended no-go and G2–G5 remain unchanged. No new supported envelope,
quality improvement, release approval, paid call, dependency install or Docker
action is claimed.
