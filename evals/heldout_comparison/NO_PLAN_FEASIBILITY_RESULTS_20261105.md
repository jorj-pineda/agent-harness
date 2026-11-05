# No-plan feasibility stopped: backend load timeout on first request

The one authorized direct/no_plan attempt stopped on its first POST: the owned
Ollama backend returned HTTP 500 with `timed out waiting for llama-server to
start - `. No complete model response, tool call, edit or post-response residency
snapshot was recorded. Aggregate prompt/completion usage is unknown. This is a
backend startup failure, not evidence that the no-plan toolset improves or harms
coding. The attempt is closed without retry.

Original external acceptance and a separate copied-workspace regrade reject the
unchanged artifact: **0/1 accepted and 0/1 completed-and-accepted**. Those counts
retain this failed attempt; they must not be interpreted as a model-capability
score or a completed comparison.

## Authority and configuration

[PR #363](https://github.com/jorj-pineda/agent-harness/pull/363) merged at
`fa364cf30047c69ce1953e5dd17f5d58dc5c2ac3` after exact-head reviews of
`98480a8d8e65c47782a2492575695a9a23bdd806`, applicable local gates and both CI
checks. The initial freeze-binding objection was resolved by an independent
recheck: the recipe reads the exact reviewed Git object, with manifest/preflight
bindings. Scope B executed the retained-evidence verifier and inspected private
sources. Fresh check-only passed and selected hashes matched before the actual
parent issued its bound receipt under Jorge's next-five-PR authorization.
Private provenance records these prerequisites; JSON assertions and Git consistency
alone do not authenticate authority.

The [freeze](NO_PLAN_FEASIBILITY_FREEZE_20261104.md) and
[protocol](NO_PLAN_FEASIBILITY_PROTOCOL.md) selected only fresh original ordinal 2:
ho_iso_week / harness / repeat 1, direct prompt, no_plan toolset, keep-alive 300
seconds. Fifteen other offline captured fixtures were preservation inputs, never
planned live attempts. Prior cohorts remain closed; these retired-v1 tasks are
known development material, not held-out evidence.

Clean retained runtime: `db6a14b9db4ead6dd52394291ece1cc230783481`.
Ollama 0.33.2/gemma4:12b used the pinned image digest
`4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`.
Original criteria and budgets remained: context 16384, thinking off, temperature
zero, 150 dispatch seconds, 32K total/4096 output tokens, 12 tools, 6 iterations,
request cap 2048/min 256. Keep-alive 300 is not the dispatch limit. No warmup,
control, second attempt, resubmission, manual repair, download/install, Docker,
hosted/paid inference or user-service/configuration/identity management occurred.

## Recorded failure

| Planned cell | Ledger | Artifact | Runtime outcome |
|---|---|---|---|
|2: ho_iso_week / harness / repeat 1|started; cohort stopped|unchanged, rejected; exit 1|error:ProtocolStop; incomplete; transport|

The result was preserved before cleanup, but the callback raised ProtocolStop;
therefore the ledger retains `started`, not `recorded`. There is one retained
result and zero unrun planned cells. The command exited 1. Neither ledger state
nor native failure is silently rewritten as successful completion.

One 8,767-byte request was delegated. Its retained native record reports HTTP 500
and a 58-byte JSON error response. The backend log records load failure with the
same startup timeout and one POST /api/chat returning 500. The error body is
complete as an HTTP response but is not a complete native model response with
done=true or usage. The audit's `incomplete_responses` list contains that error
file; it does not indicate a cancelled sixth request or a successful token prefix.
There is no native generation completion, usage total, tool trace or loaded-model
observation to compare. We did not attempt to restart, warm up or diagnose by
additional generation.

The command ran from **2026-11-05 01:43:59.258551 UTC** to
**01:51:54.253738 UTC**, outer **474.995187 seconds**. The result records dispatch
latency **12.350485 seconds**. These are different recorded timing scopes; the
available evidence does not explain the entire elapsed-time difference. Setup,
loading, grading, preservation and cleanup are not a hard end-to-end deadline.
We do not infer a speedup, effective cache reuse or a precise host-load cause.

## Preservation and closure

The post-run audit verifies clean runtime/catalog seals, selected runtime and
dependency pins, archived capture and exact first wire bytes. All fifteen unused
fixtures verify and remain unadmitted. Original fixture bytes/modes are unchanged,
and there are **no added files** in the submitted workspace. Result change lists
are empty; this audit separately checks literal inventory. Original-criteria
regrade on a disposable copy returned 1 and is not a new model attempt.

Owned cleanup records settled/passed true, forced false, returncode 0 and no errors.
The audit observed its process group absent and port 11447 bindable. A read-only
post-exit listener check observed no listeners on 11434 or 11447. These are bounded
historical observations, not continuing ownership, backend/model attestation or
process isolation. No-go, 8-of-10 and G2–G5 remain open. No default adoption,
causal/equivalence/held-out quality claim or further inference follows.

Retained root:
`/Users/jorge/.local/share/agent-harness/trials/no-plan-feasibility-zvf5eu74`.
All 25 indexed command, receipt, native, backend and audit files match their
selected hashes. WORK_LOG.md and READINESS_HANDOFF.md retain the closure and
review handoff. This is a documentation-only results change; no fresh full-suite
run is claimed. Preparation's scripted acceptance remains mechanics evidence.

| Retained evidence | SHA-256 |
|---|---|
| `results-evidence-index.json` | `022a487af913d7e23c8c45106fdaa2ddfa076032dd472e2c059cfaf2b08b0251` |
| `execution-result.json` | `ccd10aeb592a0bef5ef6280b9650bd763e5e5c5a36b8359786463a6d2977d855` |
| `candidate/run/report.json` | `ee5525a693c75551c376d83bf9ba24dffdb86edf5198872c5cf908ef326c7bf0` |
| `results-audit.json` | `167dafde1afcf2341f534348b5e24bd2db9f0aa325641ac06a2eac712021a1e4` |
| `regrade-02.txt` | `90fbd44951026953d9c5f50303d15a0cae7b281a5f72b9cc1d8a309d9d211c1e` |
| `audit-results.py` | `1d0fdbe55f36faf3d982f08f23ab38865bc894f6beb32cccea3bb95733fa3f44` |
