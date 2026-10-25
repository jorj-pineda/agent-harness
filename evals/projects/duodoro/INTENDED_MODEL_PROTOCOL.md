# Intended-model comparison protocol

**Prepared, not executed; hosted access is blocked.** This
[unrun plan](intended-model-plan.json) scopes a future comparison of the two already
used intended models, local gemma4:12b and OpenCode Go longcat-2.5-preview-free,
on original public Duodoro T02/T08 tasks. It does not reopen any prior cell or
infer that changing the model will solve the latest provider timeouts. The
[offline boundary comparison](TIMER_REQUEST_BOUNDARY.md) found no unexpected wire
schema/options/read-content drift after an explicitly diagnostic path substitution;
backend cause remains unknown. A matched new comparison can describe each intended
model/provider configuration, without claiming a same-model harness intervention.

## Current access boundary

On 2026-10-25 UTC the official [Go documentation](https://opencode.ai/v2/docs/console/go/)
listed LongCat 2.5 Preview Free with free input/output/cache-read pricing for a
limited time. This is a public pricing observation, not proof of this account's
access, remaining allowance, exact served model revision or billing.

An authenticated **read-only GET /inference/go/v1/models returned HTTP 401** using
the existing Go credential. No inference ran; neither credentials nor the response
body were retained. The check does not establish whether the cause is the credential,
endpoint, membership or another access condition. A key's presence or a public
listing cannot clear this blocker. No credential, subscription or billing setting
was changed, and no alternative credential/model/paid route was tried.

**No cell, including local cells, may dispatch while hosted access is unverified.**
Resolve the 401 through a documented read-only access check, verify the exact
previously used model ID and current declared-free pricing/account limits, then
pin safe receipts in the actual execution freeze. Do not use a generation warm-up
as connectivity validation. If read-only access cannot be established, keep the
cohort unrun; do not silently substitute a paid or different model. Additional
paid-call budget is **$0**, with no subscription upgrade or fallback authorization.
Recheck declared-free pricing before the cohort and each hosted dispatch. Unknown
or nonzero pricing blocks work. Report billing as unobserved when it is unobserved.

## Tasks, models and order

Pin Duodoro source `5324cabba4ad3c8ad2e1cd1480218cf087d9b741`, original requests,
full-source/setup bytes and modes, reference files and independent acceptance.
T02 adds final flooring while preserving coercion/default/clamp/Infinity/ID/mode/
other-parser behavior. T08 extracts/reuses boundedDuration without flooring;
it still requires the original independent source-review rubric for helper reuse,
removed duplication and unrelated preservation. Unknown review is not acceptance.
Only server/payloadParsers.js may change in either task. No new hint or easier
criterion is introduced. Both tasks use actual CLI in both models, explicitly
separate from the original panel cohort and from broader G2 recovery validation.

| Order | Cell | Task | Configured model |
|---:|---|---|---|
| 1 | T02-G1 | Timer normalization | gemma4:12b |
| 2 | T02-H1 | Timer normalization | longcat-2.5-preview-free |
| 3 | T02-H2 | Timer normalization | longcat-2.5-preview-free |
| 4 | T02-G2 | Timer normalization | gemma4:12b |
| 5 | T08-H1 | Duration refactor | longcat-2.5-preview-free |
| 6 | T08-G1 | Duration refactor | gemma4:12b |
| 7 | T08-G2 | Duration refactor | gemma4:12b |
| 8 | T08-H2 | Duration refactor | longcat-2.5-preview-free |

Use one new literal disposable workspace path across all eight cells, one stable
CLI owner and independently absent stores/inputs with empty facts per cell.
Fresh complete initial trees must match original criteria/setup before each cell.
Never use/reset Jorge's checkout or import a repaired artifact as a starting point.
Before replacing this owned root, verify full archived-tree equality, unchanged
inputs/criteria, exact saved/restarted records, grades/regrades and completed
cleanup. Later/external edits or missing proof block staging. Preserve all fixed
rows, including blocked/unrun rows; a partial cohort is not an equal-cohort ranking.

## Shared runtime and explicit differences

The provisional published runtime candidate is
`2496f22f302143bb96bf54c87b70deb7a42f639e`, containing bounded compatible recording.
The actual freeze must pin a clean published execution revision and actual source/
asset/module/environment hashes; this candidate is not an execution freeze.
Both profiles use compact/full shared prompts, tools, memory scope, checks and
budget math. Select the existing named profile so only its intended provider is
constructed. Credentials stay outside profile definitions, public settings,
model-visible messages and published traces.

Gemma explicitly requests num_ctx 16,384 and think=false. Go uses the existing
compatible transport: its own agent-harness User-Agent, a stable fresh API-session
header per conversation and zero SDK retries, without invoking OpenCode's agent.
No thinking control is sent to Go; served generation/context defaults and model
revision remain unknown where the backend does not attest them. The harness cap
is not backend compliance. Profiles/actual configurations are different model+
provider pairs, not an isolated model-weight change.

For each task, actual initial normalized messages, tool schemas, temperature and
output allowance must match across the two providers and repeats at the shared
literal root. Per-provider initial body bytes must match across repeats. Per-conversation
routing headers may differ outside the captured body; validate routing without
publishing header values. Native Ollama
and compatible bodies differ by format/model/options and are **not byte-equivalent**.
Freeze each exact provider-specific wire body/hash and rejection guard; never
rewrite bodies or drop fields to manufacture experimental equality. Later histories,
tokenizers, load/caching and sampling may diverge. Record those limits explicitly.

## Fixed limits and failure policy

Every cell keeps 300 dispatch seconds, 64,000 total reported tokens, 8,192 output
tokens, 24 tools, 12 iterations, request output 2,048/minimum 256, context 16,384,
ratio 3 and temperature 0. Keep one completion retry, identical-call limit 2,
file budget 2, required original pytest adapter, checkpoints on, and plan gate/root
injection/support/inspection off. Provider request timeout 180 is common experimental
configuration, not a new production default. CLI inactivity 600 does not extend
dispatch time; setup/final review/storage/cleanup may exceed it.

Eight cells reserve at most 2,400 dispatch seconds, 512,000 reported tokens,
65,536 output tokens, 192 tools and 96 iterations. Four cells per model reserve
half of each total. There is no transfer, third phase, warm-up, private failed-input
replay, follow-up message, manual repair, hidden retry, replacement or extra model.

Any **hosted provider/account/rate/limit/pricing error stops all remaining coding
dispatch**, retaining the failed or blocked cell and all later unrun rows. No
fallback or alternate key/route is allowed. Local provider failures remain failed
outcomes; continue the fixed order only when complete input/outcome/preservation/
cleanup evidence and every hosted gate still pass. Missing recording/review or
uncertain cleanup stops staging regardless of model. Uncertain delivery permits
only original-key GET inspection, never a new coding submission.

## Recording, grading and cleanup

Use equal bounded private normalized/exact-wire recording in both models, with
12 requests, 1MiB per file, 8MiB total and 0700/0600 permissions. Raw inputs remain
plaintext local evidence, not redacted content or proof of delivery/generation.
The compatible helpers preserve conversation routing through capture/first guards;
actual selected-factory preflight must still prove lazy capture allocation and
GET-only restart without recreating/overwriting input directories. Enforce first
normalized/provider-wire mismatches before forwarding and missing pairs before
further staging. No raw credential/header/account data enters public reports.

Retain all responses or failures, durable budget/progress evidence, bounded diffs
with omissions, full trees and independent grade/regrade results. Failed/cancelled/
unreported usage stays unknown; known prefixes are not totals. Absent final
TurnResponse/session history stays absent, not invented completed/check evidence.
External behavior plus unchanged protected bytes/modes determines artifact
acceptance; T08 also needs the original independent source review. Model prose,
filenames, visible checks and helper-name overlap do not establish acceptance.

After each cell, stop its owned API and close its selected client; unload only
local Gemma loaded for the evaluation, preserving existing user services/models.
Fresh owner-scoped GET restart must return identical saved evidence without
inference/new input files. Keep all operator corrections and rescue counts.
Remote inference/billing cancellation, hard return deadlines, external-writer
coordination and host isolation are not guaranteed by path checks or dispatch limits.

## Preparation and remaining gates

Fresh full T02/T08 initial fixtures fail visible/external checks; references pass
both. Requests, criterion/reference/setup hashes/modes match the original plan,
and Duodoro head/status/index/refs fingerprints stay unchanged. Four provisional
shared prompt/registry/profile/allowance contracts (two tasks × two models) match
per task and yield allowance 2,048 at one owned provisional path, with empty facts.
They use no constructed provider or SDK/network request. A dummy validation string
was never sent or treated as a credential/access proof. Actual first inputs,
execution roots/stores, source factories, SDK serialization and client preflight
remain unset. 78 existing profile/criterion/compatible-recorder regressions passed.

Recipe/receipt:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-intended-model-plan-pcmtt829`

The plan pins their hashes. These are offline preparation checks, not G0 on an
actual execution candidate or a client/model result. Next requirements are current
free/access/account receipts, fresh actual inputs/stores/environment, actual
selected-provider scripted CLI edits/checks/failure/missing-usage/cancellation/
restart flows, routing/privacy/bounds and no-forward/missing-review/staging negative
checks, exact runtime full offline/static gates and both scripted matrices with
labels/limits inspected. Independently review and merge the exact execution freeze,
then recheck every launch gate before any named one-pass cell. No old helper may
overwrite retained evidence or turn a completed failure into a retry slot.

## Interpretation

Primary counts are completed-and-independently-accepted /4 per model and /2 per
task/model. Show artifact acceptance, observed check state, provider failures,
regressions, interventions, unknown usage, elapsed scopes and cost limits separately.
Publish raw rows and variation, including unrun/blocked rows; no aggregate alone
establishes a useful envelope. Two temperature-zero repeats on two selected public
tasks are not the roadmap's held-out evaluation across 20–30 tasks with repeated two-model measurements or a
causal ranking. The original 8/10 requirement and all prior cohorts remain intact.

This completes only the boundary/recording/protocol-preparation three-PR batch.
**No inference is part of this PR or authorized by merging it.** The existing access
blocker and separate execution preparation/review remain. Original no-go/feature
freeze, production defaults and G2–G5 remain unchanged; no release is approved.
