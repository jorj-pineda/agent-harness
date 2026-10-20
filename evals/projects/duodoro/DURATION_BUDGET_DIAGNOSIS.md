# Duration completion budget diagnosis — 2026-10-20

Both treatment duration patches in the [feedback comparison](EDIT_FEEDBACK_RESULTS.md)
passed external acceptance but stopped before a check or final answer. Offline
replay reproduces the recorded stop without changing the budget gate: each run
had **11,057 tokens remaining**, while the next prompt's byte heuristic alone was
**12,762 tokens**, plus a minimum output reservation of **256**. The shortfall is
**1,961 estimated tokens**. This is a total-token stop, not a wall deadline,
provider failure, iteration limit or a failed check. No `run_command` was dispatched.

The [metadata-only replay observations](../../results/2026-10-20-duodoro-duration-budget-diagnosis.json)
cover both fixed runs, not new attempts. No inference, transcript reduction,
instruction removal, patch repair or budget increase occurred. The two accepted
artifacts remain `budget_exhausted` / `not_run`; they are not completed workflows.
The original ten-task and twelve-attempt results and release gates are unchanged.

## What the recorded transcript contains

Both runs follow the same tool-name sequence: read root `AGENTS.md`, list the
library directory, read `format.ts`, read `format.test.ts`, emit a plan, submit a
rejected escaped replacement, then submit the successful exact replacement. Seven
assistant responses are retained. The next request is prevented before inference;
no saved response establishes that Gemma would next choose verification or finish.
Matching sequence and aggregate numbers do not establish deterministic replication
or identical rendered requests; the original workspace-path limitation remains.

| Evidence per run | Observed value |
|---|---:|
| Reported prompt tokens | 51,679 |
| Reported output tokens | 1,264 |
| Total used / configured cap | 52,943 / 64,000 |
| Remaining total allowance | 11,057 |
| Final prompt heuristic / minimum output reservation | 12,762 / 256 |
| `AGENTS.md` tool-result payload | 17,396 heuristic bytes |
| Requests containing that tool result | 6 |
| Repeated exposure of that payload | 104,376 heuristic bytes |
| Tool-schema estimate per request | 2,541 tokens |

The instruction tool result is **57.09% of final message-payload bytes**, excluding
schemas and message-overhead tokens. Its 17,396 bytes count the encoded tool-result
content plus tool name according to `message_size_bytes`, not raw source bytes or
provider wire bytes. The directory listing adds 4,003 payload bytes and is resent
in five requests. These measurements locate large recurring inputs; they do not
attribute a percentage of actual backend tokens, cost or failure causality.

The seven attempted requests' heuristic prompt estimates are 3,129; 8,946; 10,299;
10,555; 11,127; 11,431; and 12,027. These are **not per-request reported usage or
anchored gate estimates**. Full prompts and schemas are resent at the normalized
provider boundary; backend caching/accounting is not independently observed.

## Replay and evidence limits

Replay uses the clean pinned treatment runtime
`5974d2201cb12829de2b593b602c649515e98b0d`, its actual schemas and saved settings,
run exports and finalized session archives. It verifies the freeze, run/settings
hashes and schema identity before sizing. It builds a registry with an empty
temporary FactStore but invokes no tool, model or configured turn. No original
store, fixture, criterion or source checkout is modified.

The reconstructed final transcript and known aggregate usage are passed to the
original `TokenBudget.request_allowance`. With no reconstructed reported anchor,
the byte heuristic alone produces the **exact saved completion reason** in both
runs. This proves the conservative heuristic suffices to reject this request;
it does not establish what a tokenizer or a different anchor would permit.

The API archives do not retain per-request reported usage. Aggregate totals cannot
recover those counts, per-request output allowances or every anchored estimate.
Those fields remain null. Do not calibrate a tokenizer ratio from the seven
heuristics as though they were seven reported counts, or claim that removing
1,961 tokens would preserve the model's choices and complete the workflow.

The local recipe and raw inputs remain separate from the public metadata:

`~/.local/share/agent-harness/diagnostics/duration-budget-20261020-46o15xgr/`

The recipe hash is in the observations. With the original local comparison bundle
still present, run the pinned interpreter with `PYTHONPATH` set to that bundle's
`treatment` checkout, then pass `replay.py --output <new-absolute-json-path>`.
The recipe rejects an existing output path and dirty/wrong pinned runtime. Two
offline executions produced byte-identical observations. This is not a portable
full-source/trace bundle. Old API records do not acquire PR #154's new fingerprints.

## Disposition

Keep the preventive gate and approved limits. Silently omitting applicable
repository instructions, certifying an unchecked accepted patch, raising budgets
or continuing archived history would hide the observed failure. This investigation
ships no context algorithm, tokenizer calibration, automatic verification or prompt
change, and establishes no completion/quality gain.

The next implementation should retain compact per-request usage and budget-stop
evidence for real client runs, including failed requests' unknown usage, before a
new context experiment. A subsequent experiment must predeclare its retained
instruction/check/task context, workspace identity and intervention, freeze inputs,
and measure independent acceptance plus completion under the same limits. The
historical retired context trial did not demonstrate a gain; reusing its idea is
not evidence that compaction will work here. Real reversal, held-out/two-model
measurement and the broader release criteria remain open.

Validation: both final gates reproduced exactly; source/run/settings/schema hashes
matched and repeated replay output matched byte-for-byte. The existing context/
trace focused regressions passed **27 tests**. This documentation/data-only PR
does not rerun the full offline suite or scripted matrices; their earlier results
remain historical mechanics evidence. No paid calls, extra models, Duodoro
working-checkout edits, Docker actions or model processes were involved.
