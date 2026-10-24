# Provider-timeout cohort: truncated responses

The retained T05, T06 and T10 stops reproduce through the frozen `e2a0d46`
loop using recorded normalized responses and canned tool results. All **16
request identities, complete normalized inputs and output allowances**, **32
budget observations**, and final normalized messages match. This is an offline
mechanics replay, with zero inference, real tool execution or workspace edits.
[Structured evidence](../../results/2026-10-24-duodoro-truncation-diagnosis.json)
pins the local recipe, input report and runtime source hashes.

| Task | Final request (one-based) | Saved prompt estimate | Output allowance / reported final output | Final content bytes | Canned tools before stop |
|---|---:|---:|---:|---:|---|
| T05 | 5 | 7,425 | 2,048 / 2,048 | 0 | tree, read, plan |
| T06 | 7 | 7,506 | 2,048 / 2,048 | 7,259 | tree, read, plan, write, check |
| T10 | 4 | 5,899 | 2,048 / 2,048 | 0 | tree, read, plan |

All three final responses retain native terminal observation `length`, present
prompt/output usage fields and zero normalized tool calls. The saved request cap
was 2,048, below the cumulative 8,192-output allowance. Neither the 16,384 context
gate nor the 64,000 total-token estimate gate prevented these requests. The loop
correctly stops `incomplete` on truncation without another completion retry.
It never treats the blank T05/T10 replies or T06's partial text as successful
completion. Cumulative totals and observed check states match the original run.

T05 and T10 had no edit or check to execute from the final normalized response.
T06 had already written an invalid TypeScript artifact and observed a failed
check; its later length stop did not cause that earlier bad edit. The partial
text is retained locally, not published as a proposed patch or executed as code.
No raw response body, hidden reasoning, incomplete upstream function arguments
or backend tokenization is available. Zero visible bytes do not imply zero model
generation, and reaching the cap does not prove that a larger allowance would
produce a correct patch within the unchanged dispatch budget.

Each replay verifies all **360 original outcome-evidence hashes** before and
after, reads SQLite with `mode=ro`, and confirms unchanged database bytes. Timing
and tool results are simulated; the recorded call durations are preserved as
metadata, not reproduced waits. Normalized finish categories use saved terminal
observations and tool presence; this is not a raw-response replay. Original
cohort counts and acceptance criteria stay unchanged.

Local recipe and private evidence:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-retained-failures-k5qia1i5`

Focused existing regressions passed: three truncation/missing-output-usage loop
tests. No runtime/evaluator behavior changed, so this docs/data investigation does
not rerun the full gate or scripted matrices. No model, API, Ollama service,
Duodoro checkout, dependency or Docker change occurred.

The observed cap boundary is now explicit. A future experiment could compare an
existing targeted-edit preset against whole-file generation under the same limits,
but this diagnosis selects no setting and authorizes no replay or larger budget.
Retain no-go, feature freeze and open G2–G5; these three failed cells remain closed.
