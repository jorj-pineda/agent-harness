# Deterministic client preflight

Observed on harness `6d9fff461a0f0e2460e89334ce4492c932cc9cf5` (PR #146 merge),
2026-10-18 local time / October 19 UTC. **The exercised client paths passed.**
This is scripted-provider workflow evidence with real tools in disposable copies,
not a Gemma attempt, model-quality result, ten-task trial or final execution freeze.
No product blocker was reproduced. The final manifest, model identity and G0 at
its selected revision remain pending before inference.

## Setup and evidence boundary

- Existing `create_app(components_factory=...)` injected a temporary deterministic
  provider; no application provider or runtime behavior changed. It emitted a
  read, the published reference write, the exact visible check, then a final answer.
  The calendar provider paused on its next request after the edit had settled so
  cancellation was observable. This deliberately controls timing; it does not
  predict whether cancellation will arrive in time during real inference.
- A real Uvicorn process served the source-checkout API/panel on loopback port
  18746. CLI commands used the installed `uv run agent-harness` entry point over
  HTTP. Browser actions used the actual panel controls; no direct HTTP workaround
  replaced a client control. Selecting history is required before saved-review
  controls appear; the immediate turn envelope is a separate view.
- Fresh full pinned Duodoro archives were prepared using `prepare_task`, outside
  both source checkouts. Original/source tests and authoritative external criteria
  were not rewritten. The user's Duodoro checkout was only read with Git.
- Ordinary settings came from the approved ledger's one-turn profile. Recovery
  used its reserved-phase profile on both sides of restart. Explicit simulation:
  the injected provider used the `ollama` registry key and configured `gemma4:12b`
  identity to exercise the known-provider configuration guard. Its response model
  was `scripted-preflight-NOT-gemma`; **no Ollama client was instantiated or called**.
  Captured configured identity is not backend attestation or live-model evidence.
- Isolated SQLite resources were created in the server lifecycle thread. Support
  resources were disabled. No credentials, `.env` contents, dependency installs,
  Docker, paid calls or user-repository writes were involved.
- The actual managed check ran `python -m pytest -q test_visible.py`, whose Node
  subprocess passed. Environment inspection resolved Python/pytest to the harness
  `.venv`, Node to `/Users/jorge/.vite-plus/bin/node` (`v24.22.0`), and ripgrep to
  the configured Codex executable path. This confirms this launch environment,
  not arbitrary machines or full Duodoro/Vitest/typecheck compatibility.

## Observed flows

| Flow | Observed result |
|---|---|
| CLI T05 create/chat, captures enabled | Scripted reference modified only `client/src/lib/timerPrefs.ts`; completion `completed`, checks `passed`; exact visible check exited zero. External behavior and scope acceptance passed before reversal; criteria hashes unchanged. |
| CLI T05 single-file reversal | After grading, the operator appended a later line to the separate setup sentinel. `revert-file` reported `applied`; target bytes/mode returned to baseline. A full regular-file manifest comparison found only the deliberately changed sentinel; all other paths matched baseline. |
| CLI T05 restart review | After process shutdown/restart with the same storage, `run --json` retained the completed evidence and `revert-status` retained the applied action. Read-only lookups did not re-execute the patch or reversal. |
| Panel T04 create/chat, captures enabled | Actual workspace/capture controls, New session and Send submitted the task. The envelope showed `completed` / `passed`, tool results, modified path and bounded diff. External behavior/scope acceptance passed; criteria unchanged. |
| Panel saved review and restart | Refresh history and selecting the saved run exposed workspace/transcript/configuration reviews. They showed unchanged workspace with compared file modes, valid transcript, matching configuration and estimated context fit. After API restart and page reload, the same saved run remained inspectable with chat disabled in review mode; provider-call count remained unchanged by restart/review. |
| CLI T03 cancel/restart/fork | The submission key was visible in live CLI output; one `runs --request-id` lookup showed running. One CLI cancel request settled with `cancelled` / `not_run`, preserving the partial edit and before/after capture IDs. After actual process restart, workspace was unchanged, transcript valid and configuration matching. The pinned latest turn plus explicit uncaptured-state acknowledgment admitted one new fork with the frozen continuation input. The fork completed and checked successfully; external acceptance passed with unchanged criteria. Source run JSON remained identical after the fork. |

### Recovery accounting

The recorded source and fork execution profiles were identical: each reserved
150 dispatch seconds, 32,000 total tokens, 4,096 completion tokens, 12 tool calls
and six model iterations. Both used context 16,384, per-request output cap 2,048,
minimum output 256, request timeout 60 seconds, one completion recovery, identical
call limit two, compact/full tools, required exact verification, support/root
instruction loading off, and turn captures enabled.

There were exactly two operator-submitted phases in the final recovery exercise;
no third-phase rescue. The runtime enforces each turn's allowance; the operator
still enforces the phase count. Unused first-phase budget was not transferred.
The first phase retained unknown prompt/output totals after cancellation. The
fork's synthetic reported usage was 800 prompt / 330 output tokens. Do not add
these into a known total or call this a within-budget model success. This checks
configuration persistence and unknown accounting, not real token measurement.

## Local evidence and repeatability

Raw local evidence is retained outside source checkouts at
`/tmp/agent-harness-client-preflight-6d9fff4/`: temporary provider/server script,
isolated session/memory stores, provider-call journal, CLI outputs, acceptance
outputs, prepared fixtures and `evidence-sha256.json`. These are local diagnostic
artifacts (captures contain plaintext source); they are not a committed result
ledger or portable release bundle. The committed trial ledger remains unrun.

Final exercise identities:

- CLI T05: session `ae152ed00b13422ea036c7e7fb4ed9ac`, run
  `65cce5f50dec48c99162e71d73d82805`, reversal key
  `preflight-timer-final-revert`.
- Panel T04: session `974f2c215d1a479fb959196b45c54d1e`, run
  `4b155b2ec80247eb90bf53eecd2d5e6d`.
- CLI T03: source session `12de66ad91b84a8cb734146bcd7f4aa6`, request
  `d274ab1d48b94c7688fa31f49eb30d5e`, source run
  `98777ea9e9d24b749d1fcbef313a7d5d`, pinned turn
  `052a5e009c654a0d8c5e6dbbfef026b4`, fork run
  `70235863e96c4f9995460d8d64801a96`, continuation key
  `preflight-calendar-final-phase2`.

The CLI sequence was `chat --checkpoints`, `runs --request-id`, `cancel`, wait for
settled chat output, stop/restart the same server, `workspace-review`,
`transcript-review`, `configuration-review --message`, then `continue-session`
with `--expected-turn-id`, `--request-id` and `--acknowledge-uncaptured-state`.
Read final evidence with `run --json`; do not resubmit to discover status. The
ordinary reversal used `revert-file RUN PATH --request-id KEY`, followed after
restart by `revert-status --request-id KEY`. All commands used the explicit
loopback base URL and isolated preflight user ID.

Setup failures were retained, not silently counted as passing: the temporary
provider initially used an invalid finish label, then selected a wrong reference.
The latter produced an honest incomplete/failed visible-check result. A first
reversal was applied before the intended later sentinel edit, so a fresh copy was
used for the preservation exercise. An earlier calendar dry run buffered stdout
and recovered its key through CLI listing; the final fresh-copy exercise above
used live output and the planned keyed lookup. One startup-race CLI invocation
failed before session creation. These are operator/fixture preparation errors,
not model attempts or observed application regressions. No failed attempt was
removed from a live-trial denominator; no live trial has begun.

The 38 existing focused continuation, reversal, checkpoint and persistence tests
also passed. No application/evaluator code changed, so this documentation-only
follow-up does not claim a new full-suite baseline. The temporary server was
stopped and the listener released; no evaluation-owned inference process started.

## Remaining execution gates

### Automated workflow follow-up — 2026-10-21 UTC

The later `tests/test_cli_recovery_workflow.py` regression joins the CLI paths in
one isolated real-file exercise: chat with a simulated lost response after an
actual partial write, explicit cancellation, fresh API lifecycle, read-only
workspace/transcript/configuration review, one guarded fork with a real pytest
check, then explicit single-file reversal of the fork followed by its source.
It preserves a pre-existing target line/mode, visible test bytes and later work in
an unrelated file. A third API lifecycle reads both keyed reversal records and
the unchanged run evidence without execution. Cancellation usage remains unknown
despite a known prefix; both phases capture the same reserved 150-second/32K
configuration. The phase count is controlled by the test, not a cross-turn budget
feature. No actual Ollama adapter or inference runs: configured Gemma identity is
used only to exercise the existing profile guard. The simulated transport loss
does not measure socket timing, actual clients across process exit, or browser
controls. This adds repeatable regression coverage, not live recovery/G2 acceptance
or a model-quality result, and does not rewrite the original preflight observations.

This covers representative CLI and panel paths plus the specifically assigned
CLI recovery/reversal flows, not every task/client permutation or panel recovery
control. Existing regression tests remain supporting mechanics evidence. No new
feature is justified by this preflight. Before inference, review/freeze the final
harness revision, all ten exact requests, criterion/setup hashes, fresh workspace
manifests, settings and served model identity; run G0 there. T08 still needs
independent review of the eventual model patch for actual helper reuse.
