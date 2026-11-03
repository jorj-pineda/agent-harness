# Residency development experiment: stopped after the first cell

The authorized run stopped after the zero-second ISO-week cell reached its
150-second dispatch limit during its sixth request. Its unchanged artifact was
independently rejected. Aggregate usage is unknown, so all fifteen later cells,
including every 300-second cell, remain unrun. This closes the frozen run without
retry; **the residency comparison is unresolved**. There is no measured residency
benefit, arm equivalence, causal/held-out gain, default adoption or release closure.

## Execution and evidence scope

[Freeze PR352](https://github.com/jorj-pineda/agent-harness/pull/352) merged at
`cdd5a0b5bc794d7341339330a1b83ea7ed9369fd` after review of exact head
`789c5e2a96968b39db96914782fafefe73aa657d`, both CI checks, local gates and
independent retained-hash verification (28 selections). The actual parent issued
its bound receipt under Jorge's existing request to finish the experiment after
fresh read-only candidate preflight. Receipt provenance and the exact command are
retained. No receipt schema or Git/hash check independently authenticates authority.

Runtime: clean retained `06502c680f0702195f1aa401f9dfc3e1074ebcf9`, unchanged
[protocol](RESIDENCY_DEVELOPMENT_PROTOCOL.md), fixed sixteen-cell schedule and
original criteria/budgets. Known retired-v1 tasks are development material.
Only local Ollama 0.33.2, gemma4:12b at the frozen digest, with evaluation-owned
endpoint 11447 and a fresh per-cell APFS clonefile image was used. No warmup,
resubmission, manual artifact repair, downloads, model/dependency installs,
hosted/paid calls or user-service/configuration management occurred.

The one command ran from **2026-11-03 20:17:17.522795 UTC** to
**20:20:31.558132 UTC**, returned 1 and was never resubmitted. Its 194.035337-second
outer duration includes setup and preservation; reported dispatch latency is
150.004526 seconds. Neither is a hard end-to-end deadline guarantee.

## All planned cells

The result is the first schedule ordinal. Its ledger state remains `started`:
the executor preserved the result and settled the backend, then raised the stop
before the callback could mark it `recorded`. This does not mean the resource is
still running. Remaining cells are unrun, not failed or zero-score attempts.

| Ordinal | Task | Repeat | Keep-alive seconds | Ledger state | Result |
|---|---|---|---|---|---|
| 1 | ho_iso_week | 1 | 0 | started | rejected; wall limit; usage unknown |
| 2 | ho_iso_week | 1 | 300 | unrun | unrun |
| 3 | ho_rolling_mean | 1 | 0 | unrun | unrun |
| 4 | ho_rolling_mean | 1 | 300 | unrun | unrun |
| 5 | ho_endpoint_record | 1 | 0 | unrun | unrun |
| 6 | ho_endpoint_record | 1 | 300 | unrun | unrun |
| 7 | ho_cache_policy | 1 | 0 | unrun | unrun |
| 8 | ho_cache_policy | 1 | 300 | unrun | unrun |
| 9 | ho_iso_week | 2 | 300 | unrun | unrun |
| 10 | ho_iso_week | 2 | 0 | unrun | unrun |
| 11 | ho_rolling_mean | 2 | 300 | unrun | unrun |
| 12 | ho_rolling_mean | 2 | 0 | unrun | unrun |
| 13 | ho_endpoint_record | 2 | 300 | unrun | unrun |
| 14 | ho_endpoint_record | 2 | 0 | unrun | unrun |
| 15 | ho_cache_policy | 2 | 300 | unrun | unrun |
| 16 | ho_cache_policy | 2 | 0 | unrun | unrun |

Code artifacts: **0/1 observed attempted artifact accepted**, 0 completed-and-accepted.
Questions: none attempted. Only one zero-second attempt exists; the planned
300-second arm has no attempt. Do not publish these as 0/8 versus 0/8 acceptance
rates or substitute historical runs for missing cells.

## What the model did and why the run stopped

Five complete native responses produced five tool calls: tree, two file reads,
a malformed quoted argv refusal, then a corrected argv visible-check command.
The correction reached pytest, which failed on the original ISO-year boundary.
No edit was applied. Original `week_key.py` and visible test bytes/modes remain
unchanged; an isolated copied-workspace regrade against the original external
criteria returns 1 and reproduces rejection. There was no manual patch rescue.

The sixth POST was delegated and cancelled at the wall limit. Its retained failure
records `CancelledError` and `response_body_complete:false`; no complete response
or post-response snapshot exists for that call. Result termination and completion
status are `budget_exhausted`, with the recorded 150-second wall-limit reason.
Both aggregate prompt and completion usage are null, not zero or the prefix sum.

The ledger's exact stop reason is **`residency_observation`**, not `missing_usage`.
The frozen observation wrapper latches on any BaseException from its wrapped chat,
including cancellation, and its reason takes precedence in executor stop selection.
All five completed responses have successful not_loaded snapshots. These records
support wall cancellation with unknown usage; the ledger label alone does not
establish a failed metadata GET. The label is preserved unchanged. Distinguishing
wrapped-call cancellation from observation failure is a possible diagnostic follow-up,
not a reason to resume this cohort or reinterpret its missing treatment arm.

## Native prefix and residency observations

The five complete responses report **10,772 prompt and 131 output tokens**.
The cancelled sixth request's usage and timings are unknown. Native duration sums
below describe only the complete prefix; they are not host timing spans and do
not sum to the whole run or independently identify a latency cause.

| Native field | Five-response prefix seconds |
|---|---:|
| total_duration |136.618450833|
| load_duration |21.199149958|
| prompt_eval_duration |104.677106000|
| eval_duration |10.651094000|

All five post-response owned GET snapshots record `not_loaded` for selection 0.
Startup metadata was empty, with the selected version/digest. These point
observations are consistent with this cell's configuration but do not prove
continuous residency, cache behavior or model attestation. No 300-second residency
observation exists. The configured intervention was never tested in a live cell.

## Preservation, settlement and limits

Post-run read-only checks verify clean source, original catalog/criterion seal,
selected runtime/dependency pins and captured inputs; the submitted first wire
bytes match the archived capture exactly. All fifteen unrun prepared fixtures
verify and remain unused. The original cell files are unchanged, but final
inventory equality is **false**: visible pytest generated five files beneath
`.pytest_cache`. The cache is retained without reset; added/modified/deleted
result lists are empty under the existing change-tracking policy and do not imply
literal inventory equality.

Owned cleanup records `settled:true`, `passed:true`, `forced:false`, returncode 0
and no errors. The post-run audit found the owned process group absent and port 11447
bindable. No listener was observed on 11434 or 11447 after exit. These are bounded
historical observations, not a continuing resource lease or host isolation.
Original acceptance regrading used a separate disposable copy, not another model
attempt. The read-only audit and results documentation make no model-quality or
release-readiness claim. Original no-go/8-of-10/G2–G5 remain open.

Retained root:
`/Users/jorge/.local/share/agent-harness/trials/residency-dev-2dan9i1e`.
`EXECUTION_LOG.md` records gates, approval, command, stop and audit. The evidence
index pins 44 result/command/native/observation/backend/audit files, with the index's
own hash recorded below. Model image bytes and source fixtures remain selected by
the original freeze and runtime seals rather than duplicated into this index.

| Retained evidence | SHA-256 |
|---|---|
| `results-evidence-index.json` | `494762959ee4e59dbf50581f1efabd0f52f8e970a73d34a5bad5b92dfb194f3f` |
| `execution-result.json` | `c1f8e1526524482e27ba9473364015dc4489f59ca2febbeddb701da6eb4ad604` |
| `candidate/run/report.json` | `9ab83d29af84224495ba65037d3b4c146878c691207de5ce90bf61278ec9512c` |
| `results-audit.json` | `38fb3a88c085cf99b47bfb59d851386bf36ceb6d47d698767149507bfb8968d3` |
| `regrade-01.txt` | `234a41506e325069acb309a005dcc44ced1b4c1854c1c7845e446d6bb7d6a344` |
| `audit-results.py` | `5b0f8b5ac674a465893439bc64c9a2da6df0a15241e3628b6a3cc38eeed1b61a` |
