# Whole-file cohort: three offline failure investigations

These investigations explain retained observations from the
[ten-task cohort](WHOLE_FILE_TEN_TASK_RESULTS.md), not new coding attempts.
Runtime stays `d1e5778`; its original tasks, artifacts, criteria, settings, stores
and **2/10 accepted / 2/10 completed-and-accepted** counts remain unchanged.
No inference, service startup, paid calls, timeout/budget change or patch repair.

## 1. Five provider read timeouts

[Structured metadata and recipe hash](../../results/2026-10-22-duodoro-timeout-diagnosis.json).
All five failed runs retain an untruncated budget journal ending in a request then
request failure, `read_timeout` metadata, and untruncated tool-progress observations.
Their last completed tool was `emit_plan`, without an observed tool error.
No source file changed. The final response is absent, and their durable session
archives contain zero messages/turns; exact request reconstruction is unavailable.

| Task | Failed request (one-based) | Prompt-token estimate | Output allowance | Last tool end → run finish |
|---|---:|---:|---:|---:|
| T02 | 6 | 14,979 | 1,405 | 60.02 s |
| T07 | 6 | 10,201 | 2,048 | 60.04 s |
| T08 | 5 | 7,747 | 2,048 | 60.04 s |
| T09 | 5 | 7,104 | 2,048 | 60.04 s |
| T10 | 4 | 5,953 | 2,048 | 60.03 s |

These intervals use persisted tool-end/run-finish timestamps; they include any
request preparation and finalization between them. They are not exact failed-call
latency, pure inference duration or time-to-first-byte. Each request passed the
configured token/context gate. The common roughly 60-second interval is consistent
with the configured HTTP read timeout; neither a token-gate stop nor a proven
backend root cause follows from it. T02's smaller output allowance is retained.

The pinned Ollama adapter sends `stream: false` and maps `httpx.ReadTimeout` to
safe `read_timeout` metadata. A separate offline MockTransport probe for each
recorded allowance reproduces that classification with one call and no retry,
and verifies the configured 60-second read timeout/non-streaming request shape.
It raises immediately with a placeholder message: **scripted transport contract,
not actual-request replay, reproduction of a 60-second wait, or model quality**.

Missing evidence stays missing: no raw failed response, partial output, backend
generation/queue metrics, request-start/wire timings or backend logs were retained.
The prior plan does not prove that the model was generating `write_file`, and
file size/output allowance cannot prove truncation or timeout cause. Unknown
in-flight usage stays unknown. Increasing timeouts or output limits would require
a separately scoped comparison; these observations do not establish a fix.

Raw metadata, read-only recipe and subsequent investigation outputs remain at:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-whole-file-diagnosis-hgezejz5`

The recipe verifies every original evidence-file hash before and after inspection;
SQLite is opened read-only. This diagnosis adds evidence without changing the
failed archives or treating restored history as permission to resume.

## 2. T05 missing-usage stop

[Recorded replay](../../results/2026-10-22-duodoro-missing-usage-diagnosis.json)
feeds the five saved normalized responses and recorded usage through the pinned
`run_turn`, with the exact captured tool schemas and canned saved tool results.
All five normalized request identities/allowances, all eleven budget observations
and the final normalized message history match the retained run. Timing and tool
execution are simulated; no real workspace tools, provider HTTP calls or inference.

The first response was empty but reported usage; the loop used its one permitted
completion-recovery message. The next responses invoked tree, file read and plan.
The fifth normalized response was again empty, had no tool calls, and had unknown
prompt/output usage. The configured per-request output gate stopped it with the
same `incomplete` reason: “Provider did not report output-token usage; request cap
cannot be checked.” No write was dispatched, no check ran, and aggregate usage
remained unknown. The known first-four response prefix is 14,761 prompt / 392
output tokens; it is not the full task total.

This is a reproduced preventive stop, not evidence that the gate caused an
otherwise correct patch to fail. In the saved final response there is no proposed
write to execute. The pinned adapter maps absent `prompt_eval_count`/`eval_count`
to unknown; raw payloads are unavailable, so absent versus explicit null fields,
the original wire finish metadata, and the cause of the empty response remain
unknown. Replay reconstructs finish categories consistent with observed loop
flow, without claiming a raw-payload replay. Disabling guards, guessing zero usage,
adding retries or substituting a backend response is not supported by this evidence.

## 3. T06 visible pass and false safe-integer claim

[Counterexamples](../../results/2026-10-22-duodoro-safe-integer-diagnosis.json)
run the actual retained source in a fresh full disposable copy with the pinned
Node binary. The unchanged visible test passes again; unchanged independent
acceptance rejects it again. Full eligible manifests/modes match before and after
these checks, and every original evidence hash remains unchanged. No model,
scripted patch injection or manual repair is involved in this artifact probe.

The frozen visible adapter checks `NaN` only. The submitted implementation uses
`!Number.isInteger(state.playerCount) || state.playerCount < 1`, which rejects
`NaN` but admits unsafe integers. The external criterion already covers
`Number.MAX_SAFE_INTEGER + 1`; no criterion change is needed to expose the defect.

| Count | Integer? | Safe integer? | Actual waiting / ready | Required |
|---|---|---|---|---|
| 1, 2, `Number.MAX_SAFE_INTEGER` | yes | yes | start / again | start / again |
| `Number.MAX_SAFE_INTEGER + 1` | yes | no | start / again | null / null |
| `2 ** 54` | yes | no | start / again | null / null |
| `Number.MAX_VALUE` | yes | no | start / again | null / null |
| `NaN`, `Infinity`, 1.5, string `"1"` | no | no | null / null | null / null |

The saved answer claims positive **safe** integer handling and no implementation
issues. The safe-integer claim is contradicted by the actual source and these
counterexamples. This is a specific false claim, not a complete semantic score
for all answers. The harness's `completed` / `passed` fields report completion
and the visible check actually run; they do not certify independent acceptance.
No “verified” state is inferred. The existing external boundary did its job.

Keep the original visible test, criterion and rejected artifact frozen. This
finding does not justify weakening acceptance, treating visible pass as semantic
proof, adding a model judge or silently applying a Duodoro patch. Any later
supported-task envelope needs meaningful operator-owned checks and honest limits;
these probes do not establish that another prompt or guard would fix this model
error under the agreed budgets.

## Validation and disposition

All three tasks are complete as bounded offline investigations. The three local
recipes are hashed in their corresponding structured results. Each verified all
203 original evidence-file hashes before/after its work. Focused regressions passed:
25 provider/failure-review tests, two missing-usage loop/API tests, and 28 Duodoro
criterion regressions (**55 total**). No application/evaluator code, prompt,
default, criterion, model, budget or timeout changed. This docs/data slice does not
rerun the full offline/static gate or scripted matrices as model-quality evidence.

The cohort remains no-go; G2–G5 and the original acceptance counts stay unchanged.
The observations do not identify a backend root cause or demonstrate a successful
remedy. The next bounded implementation candidate is minimal, allowlisted provider
diagnostics for the missing evidence—such as request-settlement timing and terminal
response/usage-field presence—without retaining raw contents or changing retry,
budget or timeout semantics. That needs its own scoped design/review and offline
regressions before any separately frozen live follow-up. Alternatively, Jorge can
choose a narrower supported envelope. No new trial is implied by these diagnoses.
