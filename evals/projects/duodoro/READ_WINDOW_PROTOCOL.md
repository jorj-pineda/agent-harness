# Beam default-read-window comparison protocol

**Prepared and unrun, 2026-10-26 UTC.** The [fixed eight-cell plan](read-window-plan.json)
compares fresh `beam-free` control cells with an opt-in 80-line default read window.
This follows the [retained payload diagnosis](BEAM_PAYLOAD_DIAGNOSIS.md), not an
assumed improvement. The [earlier Gemma/Beam results](BEAM_COMPARISON_RESULTS.md)
remain unchanged and are not reused as this comparison's control arm.

## Single variable and fixed order

Both arms use the same Go model/route, application runtime and named profile name
`beam`. C explicitly sets read_file_default_lines=0; W explicitly sets 80. Other
profile definitions/settings, original task messages, temperature, tools/checks,
empty initial facts and budgets stay equal. The opt-in setting changes the production
read_file description and omitted-end result behavior together; this is the tested
interface, not an isolated numeric field with an artificially identical schema.

| Order | Cell | Original task | Default read lines |
|---:|---|---|---:|
| 1 | T02-C1 | Timer normalization | 0 |
| 2 | T02-W1 | Timer normalization | 80 |
| 3 | T02-W2 | Timer normalization | 80 |
| 4 | T02-C2 | Timer normalization | 0 |
| 5 | T08-W1 | Duration refactor | 80 |
| 6 | T08-C1 | Duration refactor | 0 |
| 7 | T08-C2 | Duration refactor | 0 |
| 8 | T08-W2 | Duration refactor | 80 |

Each arm has four planned attempts, two per task. The fixed alternating blocks reduce
one simple order imbalance; they do not control hosted revision, caching, routing or
all state/order effects. Two temperature-zero repeats on public tasks are not a
held-out benchmark, statistical evidence or a generally useful task envelope.
Eighty lines is an unmeasured experimental choice, not a recommended production value.

Initial normalized messages and output allowance must match per task at the actual
common root. Only read_file.description may differ across initial tool schemas;
parameters, other tools and base prompt stay equal. Initial request fingerprints and
exact compatible bodies intentionally differ across arms. Freeze each exact first
normalized/wire input and require equality across repeats within task/arm. Do not
substitute paths, tool IDs, state or fields to manufacture equality. Later histories
and reported usage can diverge as part of the observed behavior.

## Original artifacts and instruction evidence

Source remains Duodoro `5324cabba4ad3c8ad2e1cd1480218cf087d9b741`. Preserve exact
original request/setup/reference/external-criterion hashes and modes. Only
`server/payloadParsers.js` may change for acceptance. T02 requires flooring with
other behavior retained; T08 preserves fractions and requires the original independent
source rubric for shared-helper export/reuse/removal of duplicate code. Pending or
unsupported source review is unknown, not accepted. No extra task hint, reference
exposure, easier check or new acceptance criterion is added to live cells.

Use one new literal owned disposable workspace for all eight cells with independent
absent stores/inputs and empty facts per cell. Full initial trees match the original
contracts. Never use/reset Jorge's checkout as an evaluation fixture. Complete artifact
archive, unchanged protected bytes/modes, external grade/regrade, T08 source verdict,
identical saved GET restart evidence, input pairing and settled cleanup gate common-root
replacement. Missing proof or later/external changes blocks staging without discarding
work. Retain failed, blocked and unrun planned rows if the cohort stops.

The window limits default replies only when end_line is unset; explicit ranges can
still request more within existing hard limits. Whole-file hash/line scanning remains,
and metadata distinguishes selected content from unread ranges. LOAD_ROOT_AGENTS=false
is common to both arms; this experiment does not alter the separate whole-file loader,
summarize/drop prior history or rewrite continuation state.

Record supplementary AGENTS.md read coverage: union of successful observed line ranges,
whole-file hash/total-line provenance, and any unread ranges. Unknown or missing read
proof stays unknown. The original 265-line instruction file remains unchanged. A range
is not full-file inspection, instruction understanding or compliance; a smaller reply
is not a successful task. This observation does not alter the original acceptance
criterion or authorize ignoring instructions. Report possible evidence loss alongside
any resource difference rather than filtering unfavorable cells.

## Model, access, billing and limits

Runtime candidate is published `9f06bf19b7e361c8ec065f44df1835c5056442cc`. Actual
runtime/environment/client inputs remain for the execution freeze. Use compact/full
shared runtime and the selected Beam-only provider, Go compatible base
`https://opencode.ai/inference/go/openai/v1`, own User-Agent, stable bound session
header, zero SDK retries and explicit bounded reasoning-content replay. No reasoning
effort/thinking option is added. Served revision/defaults/tokenizer/context compliance
and caching remain unknown. No Gemma or additional model is part of this cohort.

Read-only metadata returned 200 and listed beam-free during preparation. The current
[Go documentation](https://opencode.ai/v2/docs/console/go/) declares free input/output/
cache read and unlimited Go use, limited by time and region. Jorge previously attested
active Go/account-region Beam availability. Metadata is not a documented authentication
test or proof of generation, entitlement, remaining allowance or billing. Prior returned
zero-cost responses are historical observations, not a guarantee for this cohort.
Recheck exact listing and declared-zero pricing before the cohort and each cell;
unknown/nonzero/changed availability blocks dispatch. No generation warm-up runs.

Additional paid allowance is **$0**. Existing Go credential/route only, with no new
credential, balance spend, paid fallback/upgrade, alternate model or hidden retry.
The evaluation-only post-response guard requires finite non-boolean numeric usage.cost=0,
billing_source free/go/go-plus, and estimated absent or literally false before returning
proposed tools or allowing another call. Invalid/missing/positive/estimated/balance/byok
metadata stops all remaining cells. The [billing fields](https://opencode.ai/v2/docs/console/api/inference/)
are backend reports, not invoices. Post-response refusal cannot prevent an already-incurred
unexpected charge; known returned prefixes cannot establish unknown aggregate cost.

Each cell keeps 300 dispatch seconds, 64,000 reported total tokens, 8,192 output tokens,
24 tools and 12 iterations; request output cap 2,048/minimum 256, context 16,384, ratio 3,
temperature 0, completion retry 1, identical-call limit 2 and file budget 2. Require the
original configured pytest adapter after edits. Checkpoints on; plan gate, root injection,
support and inspection off. Provider HTTP phases 180 seconds and CLI response inactivity
600 remain experimental, not changes to defaults or dispatch budget. Final review/storage/
cleanup can exceed dispatch time. Eight cells reserve at most 2,400 dispatch seconds,
512,000 reported tokens, 65,536 output tokens, 192 tools and 96 iterations, half per arm.

Any hosted provider/account/rate/quota/transport error or missing usage stops all remaining
cells. Ordinary bounded harness budget stops or response truncation remain unsuccessful
cells; proceed only after complete preservation/usage/billing/review proof and valid launch
gates. There is no transfer, follow-up coding message, repair, replacement, retry or extra
cell. Uncertain delivery permits original-key GET review only, never a new submission.

## Preparation and required execution gates

Fresh T02/T08 template validation passed initial visible/external failure and reference
visible/external success. Original request/setup/reference/criteria hashes and modes match;
Duodoro head/status/index/refs are unchanged. Four provisional prompt/registry/profile/
allowance contracts passed with no provider construction. Messages/2,048 allowance match;
only read_file.description differs across arm schemas. These are provisional contracts,
not actual selected SDK serialization or source client inputs. A separate read-only metadata
request made no generation. 103 focused regressions passed.

Local preparation recipe/receipt:
`/Users/jorge/.local/share/agent-harness/trials/duodoro-read-window-plan-5k0t4dao`.

Next prepare actual CLI/source-factory/SDK inputs, scripted real-file read pagination/
explicit override/edit/check/completion/failure/missing-usage/billing/cancellation/restart
flows, routing/privacy/no-forward/staging negative gates, and exact-runtime offline/static
G0 plus both labeled scripted matrices. Freeze exact roots/stores/settings/inputs and readable
operator/evidence hashes. Independently review and merge the exact execution freeze, then
write parent approval and recheck fresh launch gates before any named one-pass cell.
Protocol merge alone is not inference permission. No fixed-output recipe may overwrite
retained evidence or reopen completed failure. Stop owned API/CLI resources afterward;
no Docker, dependency install or user service/model changes are needed for this hosted-only run.

Primary outcome is completed-and-independently-accepted /4 per arm and /2 per task/arm;
report artifact acceptance, observed checks, failure causes, request/tool counts, instruction
coverage, captured payload bytes, estimates versus reported usage, elapsed scopes, interventions
and backend-reported cost separately. Raw bounded private captures stay local (12 requests,
1MiB/file, 8MiB/cell, 0700/0600); headers/keys/account bodies remain unpublished. No missing
usage becomes zero, no visible pass/prose becomes independent acceptance, and no partial cohort
becomes an equal-cohort ranking. Original cohorts, 8/10 target, no-go/defaults and G2–G5 stay open.
