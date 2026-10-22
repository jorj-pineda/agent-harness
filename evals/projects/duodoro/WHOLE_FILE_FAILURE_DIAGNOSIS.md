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
