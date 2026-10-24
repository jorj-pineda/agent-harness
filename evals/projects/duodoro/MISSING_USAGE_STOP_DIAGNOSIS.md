# T07: missing output usage stops before further work

The latest [ten-task cohort](PROVIDER_WAIT_TEN_TASK_RESULTS.md)'s T07 run reproduces
through the frozen `e2a0d46` loop with recorded normalized responses and canned
tool results. All **six complete normalized request inputs, identities and output
allowances**, **13 budget observations**, and final normalized messages match.
The result remains `incomplete` / `not_run`, with unknown aggregate usage and no
file edit. [Structured evidence](../../results/2026-10-24-duodoro-missing-usage-stop-diagnosis.json)
pins the recipe and source hashes. This is an offline mechanics replay, not a
new coding attempt, provider HTTP replay or inference.

## What the saved evidence establishes

The first five returned responses reported a known prefix of **25,228 prompt /
147 output tokens**. One initial empty answer consumed the single configured
completion-recovery allowance. Canned tool invocations then reproduce tree,
two reads and a plan. The sixth request had a saved 10,205-token prompt estimate
and 2,048 output allowance, within the configured 16,384 context estimate gate.
It was dispatched; this was not pre-request budget prevention.

The sixth response's allowlisted provider observation records:

- `prompt_eval_count`: **absent**, not explicit null or reported zero;
- `eval_count`: **absent**, not explicit null or reported zero;
- native `done_reason`: **absent**;
- normalized content empty and zero tool calls;
- adapter HTTP/status-check duration about **7,331 ms**.

The Ollama adapter maps absent usage fields to unknown. With no native reason or
tool call, its existing normalization falls back to `stop`; that normalized category
does not attest a native successful stop. The shared request-cap gate then returns:

> Provider did not report output-token usage; request cap cannot be checked.

The final post-response observation retains unknown cumulative usage. The known
five-response prefix is not the full run total. No write or check was proposed in
the final normalized response, and none is dispatched by the replay. A plan is not
a patch. Removing the guard would not supply a missing patch or establish known
budget compliance. No automatic retry, fabricated zero count or provider fallback
is supported by these observations.

## What remains unknown

The original raw response body was not recorded. Field-presence metadata narrows
the earlier missing-usage evidence, but does not reveal why the backend omitted
the fields, whether a message/other body member was absent, hidden reasoning,
backend generation or billing. The recorded duration excludes response parsing
and is not pure inference time. Replay timing and tools are simulated; no 7.3-second
backend wait is reproduced. Normalized responses are reconstructed from saved
history, usage and terminal observations, not decoded from an original raw body.

Each replay checks all **360 original outcome-evidence hashes** before and after,
opens SQLite read-only and confirms identical database bytes. A separate read-only
check matched all 376 frozen preparation-evidence files and 25 recipes. The local
recipe and private source evidence remain under:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-retained-failures-k5qia1i5`

Seven existing focused regressions passed: five absent/null/present observation
cases, the missing-output loop dispatch guard, and the API saved unknown-usage
post-response stop. No application/evaluator behavior changed; this docs/data
slice does not rerun the full gate or scripted matrices. No inference, service,
dependency, Docker action, Duodoro edit, criterion change or patch repair occurred.

## End of the three-investigation batch

The [truncation](TRUNCATION_DIAGNOSIS.md),
[parser preservation](PARSER_PRESERVATION_DIAGNOSIS.md) and this missing-usage
investigation are complete. They explain three different retained failure
boundaries; none demonstrates a coding-quality remedy. Keep the original
**2/10 accepted / 2/10 completed-and-accepted**, unchanged 8/10 target, no-go,
feature freeze and open G2–G5. Do not count offline replays as model successes or
replace the closed cohort with probes that passed bookkeeping checks.

The next development decision should choose one concrete mechanism to test or
explicitly propose a narrower supported task scope. Existing targeted edits are a
candidate for avoiding unrelated whole-file rewrites, but their earlier mismatch
failures and mixed results remain. These investigations do not select that preset,
raise request/turn limits, authorize a fresh live freeze or establish a useful
unattended envelope. No further inference is part of this completed batch.
