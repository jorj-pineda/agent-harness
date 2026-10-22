# Scripted browser recovery preflight — 2026-10-21 UTC

**Passed for scripted mechanics only. No model inference ran.** This exercises the
[panel protocol](PANEL_RECOVERY_PROTOCOL.md) before preparing an actual live
execution freeze. It does not close G2 or change the release no-go decision.

[Structured observations](../../results/2026-10-21-duodoro-panel-preflight.json)
retain run responses, bounded diffs, request/budget observations, reversal records,
HTTP counts and hashes of local recipes, browser evidence and audits.

## Setup and simulation boundary

- Application/UI runtime: `cd11e4c33fc89f4331b50891266a9c2fef57caef` in a detached
  disposable worktree. Fresh full Duodoro fixture from
  `5324cabba4ad3c8ad2e1cd1480218cf087d9b741`; original immutable duration criteria
  and reference hashes matched. Initial checks failed and reference checks passed.
- T3 collaborative browser, incognito profile, isolated loopback origin
  `http://127.0.0.1:18752`, panel-generated owner `panel-tsr2voyi`. Workflow requests
  came from rendered controls. DOM/localStorage reads captured identities; no
  injected API requests, app functions or storage mutations bypassed the UI.
- Actual application lifecycle, SQLite stores, admission, SSE, file tools, commands,
  continuation guards and reversal implementation. A custom components factory
  supplied a fake provider and lifecycle-thread FactStore, without constructing
  an inference adapter or support retrieval resources. An ASGI observer logged
  method/path/identity without modifying request bodies or responses.
- Fake source: read the target, write a deliberately partial published reference
  fixing `formatDuration`, then hold the third request for cancellation. Fake fork:
  write the full published reference, run the configured visible check and answer.
  A review-only fake asserts on any provider call. The scripted source does not
  inspect the test file or demonstrate model instruction following/repair choice.
- Both phases retained the proposed identical reserved half-budgets: 150 dispatch
  seconds, 32K total tokens, 4,096 output, 12 tools and six iterations, with the
  existing compact/whole-file configuration. Fake reported usage is synthetic.
  Cancellation leaves aggregate usage unknown; no real token-compliance claim.

The configured provider name is `ollama` to exercise profile matching, and saved
run identity consequently displays configured `gemma4:12b`. Actual fake response
model is `SCRIPTED-NOT-Gemma`; the final answer explicitly labels the simulation.
Configured-model display is not evidence of backend/model execution.

## Observed workflow

1. Created one capture-enabled standard session; sent the exact planned request
   once through native EventSource SSE. Observed completed, non-error read/write
   cards and the real partial file before clicking **Cancel turn** once. Saved
   outcome: `cancelled`, checks `not_run`, token totals unreported.
2. Stopped the API process and restarted with the same settings, stores and root.
   Reload submitted no work. Refreshed history, selected the source, inspected
   workspace/transcript/tool progress and configuration/context. Workspace was
   unchanged against the archive, transcript valid, configuration matching and
   context estimated to fit. These observations alone grant no edit permission.
3. Loaded the latest saved source turn, supplied the planned continuation and
   acknowledgment, then clicked **Fork and submit task** once. Used **Check run**
   to inspect its finished result: `completed`, checks `passed`; one actual write
   and one configured command. Independent external acceptance on a preserved
   final artifact passed before later edits/reversals. The source partial artifact
   failed external acceptance; its graded bytes matched the source after-capture.
4. Stopped the fork API; added a declared later preservation file outside the
   coding phases. Restarted in review-only mode. Reversed the fork's target change
   once, restoring the source partial; inspected its `applied` outcome and used
   **Forget retained reversal** before reversing the source target change once.
   The second outcome was also `applied`.
5. Restarted again and reloaded. The last source reversal key restored for
   inspection without submission; **Inspect saved outcome** returned `applied`.
   Refreshed history and inspected the fork's historical completion/check and saved
   tool progress. The forgotten earlier fork key was retained in the audit; there
   is no panel arbitrary-key lookup, so it was not reopened through the browser.
6. Read-only SQLite audit found exactly two run records and two applied reversal
   records. Run records, saved sessions and response evidence remained unchanged
   (retirement flags may change). Eligible baseline bytes and regular-file modes
   were restored, the later preservation file survived, and criteria hashes were
   unchanged. Six fake calls total, no calls in review mode and no coding replay.

The panel preserves disclosure state when selecting runs. One redundant toggle
closed an already open reversal disclosure; hidden-field interaction attempts did
not submit work. A fresh reload/snapshot and normal controls completed the second
reversal. This was an automation navigation error, not an application failure.

## Validation, cleanup and limits

The 46 focused UI, cancellation, continuation, reversal, persistence and tool
progress regressions passed. This docs/data slice changes no application or
evaluation behavior; the full gate and scripted matrices were not rerun and this
is **not** the future execution freeze's G0.

All five owned API processes exited; port 18752 was closed and the owned browser
tab closed. No inference resources were started, no Docker actions taken and no
Duodoro working-checkout/index/branch edits made. Local scoped plaintext artifacts
remain at the evidence root named in the structured report; their hashes are
identifiers, not a remotely accessible evidence bundle. Screenshots were inspected
at the desktop viewport; this is not a responsive/accessibility audit.

Checks use the existing Node library adapter, not the full Next/Vitest suite.
Preservation covers acceptance-eligible files/modes and the declared sentinel;
ignored caches, ownership/ACLs, external-writer races and OS isolation remain out
of scope. Passing deterministic references proves workflow mechanics, not Gemma
coding quality or a causal recovery improvement. Historical cohorts and G2–G5
remain unchanged.

Next: prepare pinned-runtime G0 and a fresh actual panel execution manifest with
scheduled stores, workspace/owner identities, recipes/settings/environment hashes
and normalized first-request guard, then obtain independent review before live
inference. The original protocol plan remains unmodified and unrun as a live trial.
