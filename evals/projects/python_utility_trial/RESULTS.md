# Narrow Python utility trial results

**Complete cohort: 2/10 accepted artifacts, 0/10 completed-and-accepted runs.**
Both accepted artifacts are parser fixes. They passed the original external tests
and the frozen independent source rubric, but exhausted their model iterations
without native completion. Eight artifacts remain unchanged and incorrect.
This trial establishes no supported unattended envelope or release readiness.

The [protocol](PROTOCOL.md), [execution freeze](EXECUTION_FREEZE.md) and
[sanitized results](results.json) preserve the separate ten-row cohort. All ten
fresh attempts ran once in the fixed forward/reverse order, without manual patch
repair, retry, replacement, continuation or extra attempt. No rows are omitted.
The original Duodoro counts, 8-of-10 target, release no-go and G2–G5 remain intact.

## Conditions and observed outcomes

Runtime `1f64dcc1005d0bd1a457a01d52ce1bdae1990169`, local Ollama 0.33.2 and
`gemma4:12b` digest
`4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`
remain pinned. Independent two-round Sol review, both CI checks, freeze merge
`e2df5ce50147cfe653f5fa34d7022b183c5d8a64`, parent receipts and fresh launch gates
preceded generation. The actual CLI/local HTTP path used the shared application
runtime and real tools in disposable owned roots. Original requests, fixtures,
tests, references and 150-second/32K/4,096-output/12-tool/six-iteration phase limits
were unchanged. There was one fresh owned backend lifecycle per row.

| Row | Task | Native stop | Final artifact | Aggregate usage |
|---|---|---|---|---|
| PY01 | divide_zero | Iteration limit | Reject | Known |
| PY02 | slugify | 150s wall limit | Reject | Unknown |
| PY03 | stable_dedupe | Iteration limit | Reject | Known |
| PY04 | decimal_total | 150s wall limit | Reject | Unknown |
| PY05 | parse_flags | Iteration limit | Accept | Known |
| PY06 | parse_flags | Iteration limit | Accept | Known |
| PY07 | decimal_total | 150s wall limit | Reject | Unknown |
| PY08 | stable_dedupe | Iteration limit | Reject | Known |
| PY09 | slugify | Repeated-call guard | Reject | Known |
| PY10 | divide_zero | Iteration limit | Reject | Known |

Six runs hit the iteration limit, three reached finalized native wall stops and
one was blocked by the repeated unchanged-call guard. None completed. These are
settled bounded outcomes, not infrastructure-failed rows or successful workflows.
The operator completed normally with no cohort-stop marker; all ten planned rows
remain visible. Parser artifact acceptance is 2/2; every other task is 0/2.
Completed-and-accepted is 0/2 for every task, 0/10 overall.

## Acceptance and failure evidence

Original external tests and source review agree on every artifact. Both parser
implementations use a general per-line algorithm: trim whitespace, skip blank or
comment lines, split only the first equals sign and trim both components. Neither
uses example-specific outputs. Only `flags.py` changes; all visible tests and
other eligible bytes/modes remain unchanged. Both runs also record one successful
configured visible check. Passing checks did not become native completion before
the iteration cap.

The unchanged division implementation returns constant zero, the slug function
retains punctuation and separator artifacts, deduplication uses unordered set
output, and decimal totals use binary floats and truncation. These fail their
original requests and the corresponding frozen D1/S1/U1/M1 rules; direct request
violations also fail C1. Source review does not score reference-text similarity or
file overlap. It occurred after the cohort and could not guide another request.
The results PR's independent exact-content review must confirm these verdicts.

Saved tool records show rejected literal-backslash-newline replacement arguments
in the division and dedupe attempts. Several other attempts send quoted or malformed
command arrays instead of the configured check tokens. Those refusals are observed
failure boundaries, not proof that an untested prompt, encoding repair or larger
budget would help. PY06 recovered from one rejected replacement using a whole-file
write within its existing phase; this single observation is not a recovery-benefit
comparison. No corrective runtime change or further model attempt is included.

## Usage, tool accounting and timing

There are 58 paired request attempts and 55 returned response observations.
PY02/PY04/PY07 retain unknown in-flight wall-stop tails, so **cohort total token
usage is unknown**. The journal's known returned-response prefixes sum to 115,138
prompt and 3,865 output tokens; those sums include prefixes of unknown rows and
are not cohort totals, invoice evidence or proof of total-budget compliance.

The two accepted parser artifacts have known reported usage within the configured
phase bounds and saved creation-to-finalization times below 150 seconds. This is
a separate 2/10 artifact metric, not a completed workflow or a hard end-to-end
timing guarantee. All seven known-usage rows fit the token/output/request limits
and conservative tool-dispatch bound.

Raw `cost.tool_calls` is a record count, including non-executed calls. PY09 has
22 records: seven reach the executor and 15 are immediate native stop refusals.
The frozen loop sets its stop before those remaining calls can execute. Across
the cohort there are 70 tool records, including 15 such refusals and at most 55
dispatches; every phase's conservative dispatch upper bound is at most 12.
Other preconditions can refuse before dispatch too. The raw record counts remain
in JSON and are not silently relabeled as executions.

Operator wall time is 1,724.8 seconds, including repeated hash gates, setup,
snapshots, grading, restart, storage and cleanup. The 1,500-second sum of dispatch
reservations is not a hard cohort return deadline. Saved run times also include
setup/storage overhead; no dispatch-only aggregate is inferred from them.

## Independent audit and retained evidence

The final audit reconstructs all ten sealed artifacts, matches their final native
checkpoints, reproduces their executable grades, recomputes all 58 normalized/wire
input pairs and matches each first input to its reviewed row-specific pin. It
checks original-key single submission, saved run/archive equality in closed
standalone stores, GET-only restart receipts and unchanged actual store bytes.
Original double grades and both separate reconstruction-audit grades agree for
every row. The first successful audit is retained; the final audit additionally
distinguishes raw tool records from pre-dispatch refusals.

All 30 owned API/backend lifecycles exit zero with their registered processes and
groups absent and listeners free; API shutdown receipts have no unfinished tasks.
The operator exits zero and is reaped. Runtime bytes/modes, all pre-freeze cache
entries, prepared evidence, environment, owned/original model images, Duodoro
head/status/index/refs and user service/identity/configuration metadata match their
pins. User service 11434 remained absent and was never started/stopped/unloaded.
These are finite registered-resource checks, not OS isolation or discovery of
arbitrary detached workers. No working repository receives an evaluated patch.

Retained root:
`/Users/jorge/.local/share/agent-harness/trials/python-utility-live-fct828d4`.
The public JSON matches the retained final summary bytes and binds 639 outcome
files, including the executed audit, earlier audit source, result builder,
source-review verdicts, approvals and actual run evidence. The merged freeze
separately binds the 32 recipes, 6,823 preparation files and exact runtime/model
trees. Self-referential summary files and the mutable handoff journal are explicit
exclusions. No inference ran during the audits.

Five tiny public development tasks, one model and two ordered temperature-zero
repeats do not establish held-out improvement, general repeatability, causal
harness benefit, hosted behavior or a useful unattended coding envelope. This
closes the agreed protocol/freeze/results batch. It changes no defaults, budgets,
acceptance criteria, original cohort outcome or release decision, and authorizes
no further inference.
