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
