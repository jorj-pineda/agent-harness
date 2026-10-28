# Saved next-request token refusals

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

This slice saves numeric evidence only. Human CLI/panel display is a separate
client slice; JSON run review exposes the new optional metadata now. Scripted
regressions establish capture, prevention and persistence, not model quality.

Validation: 71 focused and 2,187 offline tests pass (five live deselected), plus
Ruff, formatting, mypy and both 90/90 scripted matrices with contract labels/limits
inspected. A finite differential against the pinned old budget implementation
matches 2,160 decisions across 720 states. These are mechanics checks only.
