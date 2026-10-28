# Narrow Python utility trial

**Prepared; inference unrun.** Jorge selected this narrower trial on 2026-10-28.
The [fixed plan](plan.json) is the first of three slices: protocol, independently
reviewed executable freeze, then conditional results. Protocol merge cannot launch
inference. This new descriptive cohort does not reopen any completed or stopped
cohort, replace Duodoro results, lower the original 8-of-10 target, or approve a
release. [Current release no-go](../../../RELEASE_READINESS.md) and G2–G5 remain.

## Agreed scope and fixed attempts

Use the five existing public Python development bugfix fixtures at source revision
`6be17d2ee142c0a36c4b69e73a95e558e8af43e1`: `divide_zero`, `slugify`,
`stable_dedupe`, `decimal_total`, `parse_flags`. Preserve exact original requests,
fixture bytes/modes, visible tests, external criteria and references. The plan
binds their contracts and the preparation receipts/scripts. These tiny functions
are not held out or representative of broader repository work. No features,
refactors, Node tasks, interruption/recovery workflows or second model are added.

| Order | Attempt | Task |
|---:|---|---|
| 1 | PY01 | divide_zero |
| 2 | PY02 | slugify |
| 3 | PY03 | stable_dedupe |
| 4 | PY04 | decimal_total |
| 5 | PY05 | parse_flags |
| 6 | PY06 | parse_flags |
| 7 | PY07 | decimal_total |
| 8 | PY08 | stable_dedupe |
| 9 | PY09 | slugify |
| 10 | PY10 | divide_zero |

Every row gets one fresh owned workspace and independent stores from a pinned
archive, the original request once through actual CLI/local HTTP and the shared
runtime/tools. There is one phase only: no continuation, projected fork, budget
transfer, manual repair, retry, replacement or extra attempt. A success does not
skip its predeclared second repeat. The ledger starts with all ten rows unrun;
failed, blocked and unrun rows remain in the denominator and report.

Use only local Ollama `gemma4:12b`, digest
`4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`,
expected Ollama version 0.33.2. Fresh executable/model/environment verification is
required in the later freeze; these historical pins are not a current backend
attestation. No hosted/paid comparison, new model, downloads, dependency installs,
Docker or warmup generation. The existing service on 11434 is never stopped or
unloaded. Use a fresh owned backend lifecycle per actual row and an independently
hashed APFS CoW image, not hardlinks. Freeze owned routes, ports and identity
metadata checks later; never read identity/credential contents.

## Unchanged per-attempt limits

Each attempt reserves **150 dispatch seconds, 32,000 total tokens, 4,096 output,
12 tools and six iterations**. Per-request output is at most 2,048, minimum 256;
provider inactivity is 180 seconds in all four HTTP phases, and CLI inactivity
wait is 600. Native dispatch cancellation can occur before provider inactivity.
Use temperature 0, thinking off, explicit context/Ollama context 16,384, ratio 3,
compact/full, default read window 0, one completion correction, identical-call
limit 2, file budget 2, required configured verification and no required plan.
Root instruction injection/support are off; the pinned fixtures have no AGENTS.md.
Use the exact visible check argv `python -m pytest -q test_visible.py`.

Ten attempts reserve at most **1,500 physical dispatch seconds, 320,000 total and
40,960 output tokens, 120 tools and 60 iterations**. These sums do not create a
cohort-wide pool or permit moving unused allowance. No source sharing means each
physical cost is counted once. Setup, snapshot, final review/storage, grading and
cleanup are outside a hard end-to-end deadline. Estimates may undercount reported
usage; unknown usage cannot establish budget compliance. Keep known prefixes,
unknown totals, actual allowance/cap observations and timing scopes separate.

## Independent acceptance and fixed source review

Keep original external-test acceptance, observed editable check state, completion
state, source-review outcome and final artifact acceptance as separate fields.
Original external tests are necessary but insufficient: a known floor-division
mutant passes both original division suites, while `divide(5, 2)` violates the
request for real division. This preparation control is bound in the plan. It
shows a grading gap, not a model result or universal hardcoding detector.

Every final patch receives independent source review against these frozen rules:

| Rule | Task | Required original-request behavior |
|---|---|---|
| D1 | divide_zero | General real division, including nonintegral results, and Python zero-division errors; floor division and hardcoded example results fail. |
| S1 | slugify | Lowercase ASCII slugs with punctuation/whitespace runs handled cleanly and no leading/trailing separator artifacts. No additional transliteration policy is imposed. |
| U1 | stable_dedupe | Remove repeated integer values while preserving the order of first appearances; sorting or unordered set output is insufficient. |
| M1 | decimal_total | Correct decimal-price totals with half-up cent rounding, including signed rounding; binary-float/truncation errors fail. |
| F1 | parse_flags | Skip blank/comment lines, trim surrounding whitespace and preserve equals signs within values. No unrelated parsing expansion is required. |
| C1 | All | Reject example-specific hardcoding, weakened/deleted visible tests, unrelated edits or other direct violations of the original request. Do not require reference text, filenames or implementation style. |

Review the final artifact, not a final-answer claim or filename overlap. Retain
patch/source evidence and reasons for pass/reject/unknown for every rule that
applies. Source review happens outside the coding loop and cannot guide another
model request, edit or retry. A separate exact-content review must check the
reported verdicts before results merge; no premium supervisor is added to the
product. Executable grade/regrade and preservation settle each row before the next one.
Source reviews may finish from preserved artifacts after the cohort and must be
resolved or explicitly unknown before accepted counts are published. No model
dispatch waits for a premium reviewer. Review uncertainty remains unknown, not
pass. Ambiguous requirements
must not be expanded after observing a patch. Original executable criteria stay
unchanged; this explicit added review condition makes historical raw acceptance
counts distinct from this trial's final acceptance.

`artifact_accepted` requires original external acceptance **and** source-review
pass. `completed_and_accepted` additionally requires native completed status.
Neither editable checks nor source review alone establishes acceptance. Passing
external tests with incomplete/rejected/unknown review stays provisional; unknown
counts and reasons remain visible. Independent regrade/reconstruction must agree;
no editable test may redefine authoritative acceptance. Final known-budget success
is another separate metric and requires known, in-limit phase usage.

## Preservation, refusal and stop rules

Never evaluate in Jorge's working repositories, commit/push task patches or apply
results to them. Original task/reference/criterion trees, recipes, model image,
runtime and evidence stay outside editable roots and hash-pinned. Preserve each
actual artifact and grade/regrade before proceeding; capture stores, normalized
and native-wire inputs, run/archive/check/budget/progress records and client exit
receipts. External grades and source-review verdicts never enter model inputs.
Fresh disposable roots need no restoration of an old workspace. Unknown changes,
unsupported entries, damaged protection or incomplete artifact preservation refuse.
Revalidate prior protected artifacts and exact environment before each row.

Owned API/backend lifecycles must exit normally with code zero before grading or
later allocation. Managed tool tasks/processes must be terminal and settled,
retaining their actual success, failure or cancellation outcomes; a failed check
or a tool terminated at an allowed native wall stop need not exit zero. Require
closed resources/tasks, absent registered groups, free listeners and preserved
user service/configuration/identity metadata. These gates and path allowlists
are ownership checks, not OS isolation from arbitrary detached workers or
unfamiliar code. Fresh launch gates must inspect actual processes; never assume
a prior service is stopped.

Native policy/refusal/token/context/tool/iteration results and a finalized native
`budget_exhausted` with the exact configured wall-budget reason are outcomes;
continue only after preservation, independent grades and clean settlement. An
unknown in-flight tail at such a wall stop remains unknown. `CancelledError`
alone is not a finalized wall outcome or authorization to continue.

**Stop remaining rows** on any provider/transport failure, missing/malformed
returned usage, reported request/phase overrun, unexplained cancellation, submission/
recording/runtime/model/environment/criterion drift, inconsistent grade or
incomplete preservation/cleanup. An uncertain POST permits original-key GET only:
no resubmission, new key, generic-session fallback or delayed retry. Retain the
failure, blocked and remaining unrun rows. No changed model, timeout, range,
allowance, criteria or operator repair may rescue this cohort.

## Preparation and conditional execution

Observed offline only: all five fresh initial fixtures fail visible and external
checks with exit 1; all five reference copies pass both with exit 0. Twenty
checks use disposable copies. All 42 original catalog files/modes match before
and after. The separate floor-division control confirms original test acceptance
and rejection under D1, with an explicit nonintegral witness. No provider
construction, endpoint or inference was involved. Plan values separately validate
through `Settings`. These receipts prove fixture/
criterion mechanics, not model capability or the complete operator.

The second PR must supply the full executable recipe and refusal rehearsal,
protected archives/artifacts and independent stores, row-exact actual first-input
captures through the production adapter, all-request capture guards, and a
within-task comparison allowing only each declared fresh workspace-root line to
differ. Forwarded inputs and launch hashes retain the full
paths; no input normalization is used to force identity equality. Require
actual CLI/API scripted preflight, restart GET-only review on closed clones, owned
backend/resource probes and exact-runtime focused/full/static G0 plus both labeled
scripted matrices. Simulation overrides must be explicit and forbidden in live
configuration. Bind all private recipes/evidence, original contracts, runtime tree,
model/environment/executables, ownership and the full fixed ledger in a public
freeze. No incomplete launch script is an execution authorization.

Require independent exact-head Sol review of the complete freeze and private
bindings, both named CI checks, merge, parent-observed review/approval receipts
and fresh launch/per-row gates before the third PR's conditional generation.
No old cohort approval, checkpoint or saved input grants launch authority.

Publish raw per-task counts out of two and whole-cohort counts out of ten, every
attempt and preservation/cleanup outcome, source-review reasons, observed checks,
known/unknown costs and intervention count. Repeats at temperature 0 are not
independent statistical samples; cache/hardware/timing remain uncontrolled even
with fresh owned backends. No baseline arm or second model means no causal harness
improvement claim. Even ten accepted public tiny bugfixes cannot establish the
original Duodoro target, held-out quality, broader useful unattended coding,
client/recovery reliability, default adoption, release approval or G2–G5 closure.
