# Two-task provider observation probe

Status at protocol preparation: no inference or execution freeze in that PR.
The later [launch report](PROVIDER_DIAGNOSTIC_RESULTS.md) records a setup blocker
before inference; new reviewed preparation is required, without rerunning that freeze.

After the whole-file cohort's five read timeouts and T05's missing-usage stop,
PRs #176–#178 added bounded provider observations and saved client review. This
probe asks whether the new evidence is retained on actual local-model requests.
It tests observation coverage, not a remedy, causal comparison or supported coding
envelope. Original cohorts, criteria, no-go and G2–G5 remain unchanged.

## Fixed scope

Run exactly two public tasks, in order **T05 timer_prefs, then T09
tag_filter**, once each through the actual CLI. These were respectively a missing-
usage stop and a read timeout in the retained whole-file cohort. Copy their exact
requests, criteria, references and allowed paths from
`whole-file-ten-task-plan.json`; both already have CLI assignments. Do not select
another task if either succeeds, fails or does not reproduce its historical symptom.

Use fresh full disposable copies of Duodoro revision
`5324cabba4ad3c8ad2e1cd1480218cf087d9b741`, separate empty stores and unseeded memory.
Only local `gemma4:12b`, compact/whole_file, thinking off, temperature zero; no
hosted fallback, dependency installation or Duodoro working-checkout changes.
No automatic retry, extra turn, fork, cancellation experiment or patch repair.
This is a diagnostic subset, not the original ten-task recovery protocol: T05 is
graded and reviewed after restart, with no reversal or later sentinel mutation.
Do not count it as a reversal/preservation workflow or another ten-task trial.

Per task retain the approved 300-second dispatch allowance, 64,000 total tokens,
8,192 output tokens, 24 tools and 12 model iterations. Keep 16,384 harness/Ollama
context, 2,048 request output cap, 256 minimum output, 60-second provider timeout,
one completion recovery and identical-call limit two. Keep verification required
with `['python','-m','pytest','-q','test_visible.py']`, support/root loading off and
turn captures on. CLI response wait remains 600 seconds of read inactivity;
cleanup, final review, storage and grading may exceed the dispatch allowance.
Limits are estimate/backend-report based, not hard latency/tokenizer guarantees.

## Preparation and launch gates

1. Independently review and merge this protocol. Before inference, publish a
   separate freeze at an exact clean runtime containing #178. Validate both
   initial-fail/reference-pass pairs without changing acceptance. Record source,
   criterion/reference, full eligible fixture bytes/modes, settings, runtime/tree/
   lock, recipes, command environment/executables and current backend/model identity.
2. Use separate scripted stores/workspaces to exercise actual CLI submission,
   saved review and restart with a fake provider and real tools. Cover response
   observations and duration-only request failure, plus first-input mismatch
   rejection with no forwarding. Scripted evidence must not be labeled inference.
3. Capture exact normalized first messages/schemas/output allowance using separate
   stores and the scheduled root; scheduled stores remain absent. Live wrapper
   asserts first request identity and temperature before forwarding, without
   modifying messages, tools or responses. Pin owner, paths and profile too.
4. Run focused regressions, full offline/static gates and both scripted matrices
   at the exact runtime. Inspect contract labels/metric limits. Independently
   review and merge the actual freeze before inference. Recheck all pinned inputs,
   environment and unloaded model identity before starting; drift blocks execution.

## Execution and retained evidence

Start one owned loopback API per task. Use CLI create/chat once with captures,
retain its printed submission key, inspect its run with CLI, then stop the API.
On uncertain delivery, inspect the same key; never submit a replacement. Restart
the API with a provider that rejects inference and inspect the same saved run,
human diagnostics and JSON through CLI; archive equality must hold. Keep saved
budget/progress prefixes, explicit truncation, final response/failure, checkpoints,
workspace manifests, client/server logs and all operator corrections. No raw wire
payload or backend logs are added. Stop owned APIs and unload the evaluation model.

Grade each final artifact outside the workspace before any later action. Record
acceptance, completion/check status, changed paths/modes, protected work, known
usage prefix versus unknown total, observed durations and wall timing separately.
Regrade an unchanged archived copy and verify criteria and evidence hashes.
Request failure timing describes HTTP post/status-check settlement, not pure
inference, first-byte time, backend root cause or a strict timeout guarantee.
Absent/null/present native usage states and terminal categories explain retained
responses only; a missing observation remains unknown. No backfilling old runs.

Every scheduled outcome is reported, including setup failure, drift, blocked input,
timeout, empty response, unknown usage, incomplete completion or unavailable
review. Two successes do not close G3/G4; non-reproduction does not refute historical
failures. Report whether each dispatched response/failure has the expected bounded
metadata and whether restart preserves it; record any gap as a separate blocker.
Choose the next investigation from those observations. Do not silently increase
timeouts/budgets, change prompts, repeat tasks or declare release readiness.
