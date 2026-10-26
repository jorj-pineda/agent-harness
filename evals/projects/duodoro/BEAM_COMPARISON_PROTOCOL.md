# Gemma / Beam comparison protocol

**Prepared, not executed.** The [new eight-cell plan](beam-comparison-plan.json)
uses the original public Duodoro T02/T08 tasks with local `gemma4:12b` and Go
`beam-free`. Jorge selected the cheapest model within his existing Go subscription.
This is a separate descriptive model/provider comparison, not a retry of any old
cell, a same-model harness intervention, a causal ranking or release validation.
The retired [LongCat plan](intended-model-plan.json) remains unchanged and unrun.

## Access, selection and spending

On 2026-10-25, a request sent with the existing Go credential to
`GET /inference/go/v1/models` returned 200 and listed `beam-free`. This route is
not established as a documented authentication test. A successful metadata response
does not prove generation, entitlement, remaining allowance or billing. Credentials,
headers and the response body are not published; no account settings changed.
Jorge explicitly confirmed that Go is active and Beam is available in his account/
region. Record that as user attestation, not independent Console verification.
The shared browser reached the logged-out Console login page and its owned tab
was closed without an account action.

The current [Go documentation](https://opencode.ai/v2/docs/console/go/) lists Beam
Free with free input/output/cache-read and unlimited Go usage, limited by time and
region. Public price metadata and user attestation are the prelaunch access inputs;
actual generation/tool/reasoning behavior remains unobserved until the named cells.
There is no warm-up, connectivity generation or failed-input replay. Recheck the
exact model listing and declared-zero pricing before the cohort and every hosted
cell. Unknown/nonzero pricing or changed availability blocks dispatch.

Additional paid budget is **$0**. Use only the existing Go credential and route,
with zero SDK retries and no balance spend, paid fallback, subscription change,
alternate credential/route/model or automatic replacement. An evaluation-only
response guard must require finite numeric, non-boolean `usage.cost == 0`, an
allowed `opencode.billing_source` of `free`, `go` or `go-plus`, and `estimated`
absent or literally false before returning tool proposals or allowing another call.
Missing/malformed/positive/estimated/balance/byok billing metadata stops the phase
and all remaining cells. This post-response refusal cannot prevent an unexpected
already-incurred backend charge. Retain known observations; unknown cost is unknown.
This is not new application cost enforcement. The
[documented billing fields](https://opencode.ai/v2/docs/console/api/inference/)
are backend reports, not independently verified invoices.

## Fixed cells and original tasks

| Order | Cell | Task | Model |
|---:|---|---|---|
| 1 | T02-G1 | Timer normalization | gemma4:12b |
| 2 | T02-B1 | Timer normalization | beam-free |
| 3 | T02-B2 | Timer normalization | beam-free |
| 4 | T02-G2 | Timer normalization | gemma4:12b |
| 5 | T08-B1 | Duration refactor | beam-free |
| 6 | T08-G1 | Duration refactor | gemma4:12b |
| 7 | T08-G2 | Duration refactor | gemma4:12b |
| 8 | T08-B2 | Duration refactor | beam-free |

Pin source `5324cabba4ad3c8ad2e1cd1480218cf087d9b741`, original request bytes,
full source/setup files and modes, reference and external acceptance hashes.
Only `server/payloadParsers.js` may change. T02 adds flooring while retaining all
other parser behavior. T08 preserves fractional semantics and requires the original
independent source-review rubric for actual helper reuse/removal of duplication;
unknown review is not acceptance. No extra task hint or easier check is added.
All cells use actual CLI, separate from the original panel cohort and broader G2.

Use one new literal owned disposable workspace for all eight cells, independently
absent stores/inputs and empty facts per cell. Fresh complete initial trees must
match the original contracts. Never reset or use Jorge's checkout as an eval
fixture. Full artifact/archive equality, unchanged protected bytes/modes, independent
grade/regrade/source review, unchanged saved GET restart evidence, input pairing
and completed cleanup must pass before replacing the owned root. Later/external
changes or missing evidence block staging; retain blocked/unrun rows.

## Runtime, replay and limits

The published runtime candidate is `cc7bdb4a73176592aed25e9ab58d61c55c50a793`.
This is not an actual execution freeze. Both configurations use compact/full shared
prompts/tools/checks/memory/budget math and explicit named startup profiles so only
the chosen provider is constructed. Gemma requests num_ctx 16,384 and think=false.
Beam uses `https://opencode.ai/inference/go/openai/v1`, the Go User-Agent/stable
session header recipe, zero SDK retries and explicit bounded reasoning-content
replay. Replay does not enable thinking or request a reasoning effort. Served model
revision, generation/context defaults, tokenizer compliance and caching remain
unknown unless observed; native model/provider pairs differ beyond model weights.

Initial normalized messages/schemas/temperature/allowance must match per task
across both models and repeats at the actual common root. Freeze each provider's
exact first body and no-forward guard; within-provider repeats must match. Native
Ollama/compatible wire bodies are not byte-equivalent. Do not rewrite roots, tool
IDs, reasoning or other fields to manufacture equality. Later histories can diverge.
Record reasoning-field presence, retained bytes and subsequent replay from actual
inputs/history where available, without interpreting reasoning prose as correctness.
Malformed/oversized continuation fails closed; absent state stays absent.

Each cell retains **300 dispatch seconds, 64,000 reported total tokens, 8,192 output
tokens, 24 tools and 12 iterations**. Request output is 2,048/minimum 256, context
16,384, ratio 3, temperature 0; completion retry 1, identical-call limit 2 and file
budget 2 remain fixed. Require the original pytest adapter after edits. Checkpoints
are on; plan gate, root injection, support and inspection are off. Provider timeout
180 and CLI response inactivity 600 are experimental configurations, not production
default changes or extensions of dispatch time. Setup/final review/storage/cleanup
may exceed it. Eight cells reserve at most 2,400 dispatch seconds, 512,000 reported
tokens, 65,536 output tokens, 192 tools and 96 iterations, half per model.

No budget transfer, third phase, follow-up coding message, patch repair, hidden
retry, warm-up, replacement or extra model. Any hosted provider/account/rate/limit/
pricing/billing error stops **all** remaining dispatch, including local cells.
Missing hosted usage also stops further staging, without inventing totals. Local
failures remain failed; continue fixed order only with complete preservation/input/
cleanup/review evidence and all hosted launch gates still valid. Uncertain delivery
allows original-key GET inspection, never a new submission. Preserve every planned
row even when the cohort stops early; partial results are not equal-cohort ranking.

## Evidence, preparation and remaining gates

Both models get bounded local plaintext normalized/exact-wire capture: twelve
requests, 1MiB/file, 8MiB/phase, directories/files 0700/0600. Lazy allocation must
leave existing inputs unchanged on default/review-only GET restarts. Headers,
credentials and raw account/error payloads stay out of public reports. Keep every
response/failure, bounded diff and omissions, full tree, saved budget/progress
coverage, known usage prefixes, elapsed scopes, interventions and sanitized billing
observations. Failed/cancelled/missing usage remains unknown. Initial-only archives
and absent TurnResponse fields stay absent. Do not promote visible checks or prose
into independent acceptance.

Fresh T02/T08 initial/reference validation passes its fail/pass expectations;
original request/criterion/setup/reference hashes and modes match. Four provisional
shared prompt/registry/profile/allowance contracts match per task with empty facts,
without constructing a provider. 103 focused regressions passed. The preparation
recipe/receipt are retained at:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-beam-comparison-plan-dzbd8y4j`

The plan pins their hashes. Actual first SDK inputs, stores, source factories,
client preflight, environment/model receipts and G0 remain unset. Next prepare the
actual selected-provider CLI scripted edits/checks/failure/missing-usage/cancellation/
GET restart flows and routing/privacy/billing/no-forward/staging negative gates.
Run exact-runtime offline/static gates and both scripted matrices with labels/limits
inspected. Independently review and merge the exact execution freeze, then recheck
all launch gates before any named one-pass cell. No old recipe may overwrite
retained evidence or reopen a completed failure. Stop owned APIs/clients afterward;
unload only evaluation-loaded Gemma while preserving user services/models. No Docker.

Primary counts are completed-and-independently-accepted /4 per model and /2 per
task/model, with artifact acceptance/checks/usage/failures/regressions/cost limits
separate. These public tasks and two temperature-zero repeats are not the held-out
20–30 task comparison, a generally useful envelope or release approval. Original
cohorts, 8/10 target, no-go/feature freeze, defaults and G2–G5 remain unchanged.
**No inference is part of this preparation PR or authorized by protocol merge alone.**
