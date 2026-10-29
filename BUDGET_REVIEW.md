# Saved next-request token refusals

## Validated token counts

The shared `TokenUsage` boundary accepts only nonnegative integers or null.
Booleans, numeric strings and whole-valued floats are rejected instead of coerced;
missing/null usage stays unknown, while integer zero remains valid. Ollama passes
its original JSON count values to this boundary, so malformed counts stop response
normalization before returned tool calls can execute. Other adapters validate the
values supplied by their SDKs; this cannot recover raw types already coerced by an
upstream SDK or attest backend tokenization/billing.

This intentionally tightens malformed-input handling, including validation of
saved data. Valid integer/null serialization stays unchanged; historical evidence
is not rewritten or repaired. Failed normalization retains the existing failed
request/unknown-usage behavior, not a fabricated zero total or a successful run.
No retry, allowance/default change, inference or model-quality claim follows.

## Saved request refusal evidence

New saved budget observations may include `request_block` on a
`pre_request_stop` or `post_response_stop`. It records the actual shared gate's
numbers when another request cannot fit. This addresses the distinct native C
lower-bound and P estimate stops reproduced in the
[offline diagnosis](evals/projects/duodoro/TOKEN_RECOVERY_BUDGET_DIAGNOSIS.md).
It changes no admission comparison, allowance, default, status or completion reason.

`request-budget-block-v1` contains:

| Field | Meaning |
|---|---|
| `budget` | `total` means cumulative turn tokens; `context` means the request window. |
| `basis` | `estimate` is the larger whole-prompt heuristic/reported-anchor estimate before dispatch. `reported_prompt_lower_bound` is the last reported prompt count checked after a response, before proposed tools. |
| `limit_tokens` | Configured/effective cap used by this gate. |
| `available_tokens` | Remaining total before charging a next prompt, or the full context cap. This is not an output allowance. |
| `prompt_tokens` | Estimate or reported lower bound actually used in the refusing comparison. |
| `minimum_output_tokens` | Minimum output reservation required alongside that prompt. |

At this gate, prompt plus minimum output exceeds available tokens. The record is
historical evidence, not a next-request usage measurement, backend attestation,
new budget, successful task, hard latency deadline or permission to resume.
Reported lower bounds assume retained history; estimates can undercount actual
usage. Independent acceptance and unknown source/failed-request usage stay separate.

The optional field is omitted when absent, including parsed legacy records,
non-stop events, output-cap stops, stops caused by missing usage, and reported
overruns. Those
other gates keep their existing evidence/reasons; absence must not be filled in
from today's settings or presented as zero. The existing bounded event prefix can
omit a later stop; truncation stays explicit. Old records/stores are not rewritten.
HTTP/SSE and restart review use the same runtime event and durable run store.
No payload contents, paths, tool arguments or exception text are added.

Inspect a saved run with `agent-harness run <run-id> --user-id <owner>`, or select
it in panel history and expand **Next-request token refusals (saved review only)**.
Both clients show the recorded stage, total/context budget, cap, availability
before a prompt, estimated prompt or reported prompt lower bound, and minimum
output. They validate the supported shape and recorded comparison consistency;
they do not recalculate an allowance from current settings or reconstruct missing
legacy details. The panel uses text nodes and starts collapsed. Existing saved
review stays read-only; rendering adds no request, action or stored intent.
`--json` still returns the complete original server record.

Both clients cap display at the existing 100-event prefix, label unknown or
truncated coverage and keep unsupported/malformed/contradictory details
unrecorded. No refusal in that prefix does not establish that no stop occurred.
The available count is not an output allowance: context counts window capacity,
while total counts remaining cumulative turn tokens. This view grants no retry,
continuation, new budget or automatic calibration. Old cohorts remain closed.
Scripted regressions establish capture, prevention and persistence, not model quality.

Capture validation (PR #261): 71 focused and 2,187 offline tests pass (five live deselected), plus
Ruff, formatting, mypy and both 90/90 scripted matrices with contract labels/limits
inspected. A finite differential against the pinned old budget implementation
matches 2,160 decisions across 720 states. These are mechanics checks only.

Client proof includes actual installed CLI commands over a real ephemeral local
HTTP API with scripted responses for both gate types, unchanged saved records and
workspace bytes, plus restart review through the real client/HTTP adapter and
actual panel showRun/render functions. The tested owned API exits zero, with its
process group and listener absent. Browser tooling is unavailable; Node DOM
checks cover text/render wiring, not a native-browser layout or screenshot audit.
A failed preparation probe used a nonexistent health route; its zero-call cleanup
is retained separately from the corrected successful proof. No inference ran.

Client validation: 55 focused tests, Ruff, formatting, mypy and JavaScript syntax
checks pass. The first full run had 2,217 passes and one failure at the existing
forced-process-death fixture's ten-second ready deadline, before fixture files
were created. That unchanged test passed in isolation, then a fresh unchanged full
run passed 2,218 tests (five live deselected). The initial failure is retained;
its root cause is not established and no test deadline or product behavior was
changed. Scripted matrices were not rerun for this client-only change; the
capture slice above ran both. None of these checks measures model quality.
