# Ten-task Duodoro client trial: approved preparation

**Approved for offline preparation, not yet runnable.** Jorge approved the proposed
ten tasks and limits, including the five-minute dispatch allowance and reserved
recovery phases. All ten acceptance sets now pass baseline-fail/reference-pass
validation; the seven new sets also have adversarial regression coverage. This
prepares release gate G1 in [the readiness audit](../../../RELEASE_READINESS.md).
The [deterministic client preflight](CLIENT_PREFLIGHT.md) passed the exercised
paths. Do not dispatch trial inference until the reviewed execution freeze and
G0 at its selected revision are complete. G1–G5 remain open.

The [ledger template](trial-ledger.json) contains ten unrun rows, source blob IDs,
client assignments and null results. It is documentation/data, not a new runner
or a schema the current runtime consumes. Do not fill unknowns with zero/success.

## Source, clients and scope

- Pin Duodoro at `5324cabba4ad3c8ad2e1cd1480218cf087d9b741`. Read source with Git;
  prepare a fresh disposable copy for every task. Never checkout/reset/edit the
  user's working tree. No Duodoro commits, pushes, deployment, database/auth
  services, dependency installation or Docker actions are part of this trial.
- Use local `gemma4:12b` through the source-checkout API and its shared runtime.
  Record the actual model digest, backend/runtime versions and harness commit at
  freeze time; historical digests are not evidence of the currently served model.
  No hosted fallback, additional model or premium supervisor.
- Five tasks start through the real CLI, five through the real panel. Use their
  own create/chat/cancel/review/continuation/reversal controls. Independent external
  grading may run outside the client; manual HTTP requests must not replace a
  missing client workflow. The headless wheel is not the panel distribution.
- Library-level coding only. Native Node/TypeScript checks do not establish
  Next/Vitest, TypeScript typechecking, Socket.IO, browser or full application
  regression coverage. Verify the managed command environment, not just shell
  Node. Earlier adapters were validated with Node 24.21.0; this preparation used
  Node 24.22.0. Do not infer Node 22 compatibility from that. Repository code retains host permissions.
- These are public development/trial tasks. Earlier duration/timer tasks have
  already influenced development. Neither they nor the new published candidates
  can be represented as a held-out set or ten independent benchmark samples.

## Ten tasks, in fixed approved order

Every row is a distinct request on a fresh copy of the same pinned source. In
particular T08 preserves fractional parser behavior; it does **not** build on
T02's flooring change. Existing task files/criteria are unchanged by this preparation.
New bugfix labels describe requested contracts, not confirmed production incidents.

| ID | Client | Category/request | Allowed modified paths | Preparation |
|---|---|---|---|---|
| T01 | CLI | Duration formatting: negative/non-finite values render as zero; preserve positive/fractional formatting and tags. | `client/src/lib/format.ts` | Existing `duration_format` task; historical development task. |
| T02 | Panel | Start-session duration normalization: floor final defaulted/clamped seconds; preserve coercion, defaults, bounds and other parsers. | `server/payloadParsers.js` | Existing `timer_normalization` task; historical development task. |
| T03 | CLI | Add optional Sunday-first `monthDays`, default Monday-first; preserve full grids and other exports. Planned cancel/restart/fork workflow. | `client/src/lib/focusCalendar.ts` | Existing `calendar_layout` task; no prior model inference recorded in its README. |
| T04 | Panel | Make pending-share-invite storage operations tolerate blocked/throwing browser storage. | `client/src/lib/shareInvite.ts` | Implemented and validated offline. |
| T05 | CLI | Normalize merged timer preferences before persisting them. Grade, then explicitly reverse this single file. | `client/src/lib/timerPrefs.ts` | Implemented and validated offline. |
| T06 | Panel | Reject invalid player counts in the start shortcut without changing other gates. | `client/src/lib/keyboardShortcuts.ts` | Implemented and validated offline. |
| T07 | CLI | Add matching bounded upcoming-world preview helpers in client and server rotation modules. | `client/src/lib/rotation.ts`, `server/rotation.js` | Two-file criteria/reference implemented and validated offline. |
| T08 | Panel | Extract one shared bounded-duration helper for both start-session fields without changing parser behavior. | `server/payloadParsers.js` | Behavioral criteria/reference validated; independent source-review rubric prepared. |
| T09 | CLI | Add a strict filter-normalization helper returning valid tag filters or `all`. | `client/src/lib/focusTags.ts` | Implemented and validated offline. |
| T10 | Panel | Trim and deduplicate wrap-up intention strings while preserving eligibility/order and inputs. | `client/src/lib/sessionWrapUp.ts` | Implemented and validated offline. |

### Acceptance contracts to freeze before any model run

Common to all: existing source/tests/setup bytes and modes outside the allowed
paths must remain unchanged, no unexpected additions/deletions, and allowed file
modes remain unchanged. Preserve a setup-owned pre-existing-work sentinel. Run
external checks outside the editable workspace with the existing owned bootstrap;
visible checks cannot redefine acceptance. All ten need baseline-fail/reference-pass
validation of task-specific acceptance and visible checks before the first attempt.
Include malformed/partial solutions that pass a weak visible check but must fail
external acceptance. All ten initial/reference pairs were validated in disposable
copies of the full pinned source and portable library snapshots. Incorrect partial solutions, scope
changes and tampered criteria are regression tested. These are offline criterion
checks, not evidence of model quality or completed client workflows.

- **T01–T03:** retain the exact requests in `task.json`, `task_timer.json` and
  `task_calendar.json`, their visible adapters, references and external assertions.
  T03 covers complete Monday/Sunday grids across leap/century boundaries, other
  exports and exact scope. Record criterion hashes and fresh fixture manifests.
- **T04:** keep the exact 43-character token validator and URL behavior. In SSR,
  store returns false, read null and clear no-ops. With usable storage, valid tokens
  round-trip and malformed retained tokens are removed. Throwing get/set/remove
  must never escape: store returns false, read null, clear no-ops. Do not report a
  successful store when set throws; do not silently relax validation or URL rules.
  A storage test double is adapter setup, not a live browser-storage guarantee.
- **T05:** persist `normalizeTimerPrefs({...readTimerPrefs(), ...prefs})`, preserving
  unspecified existing fields. Check bounds/snapping, invalid values, both modes,
  missing/malformed stored JSON and throwing storage. Neither input nor exported
  defaults may mutate; SSR and quota errors remain nonthrowing. Invalid supplied
  fields follow existing normalization defaults, not a newly invented merge policy.
- **T06:** `playerCount` must be a positive safe integer; reject NaN, infinities,
  fractions, zero and negatives without coercing strings/bools. Preserve valid
  one/two-player results, all other state gates and phase-to-action mapping. Native
  TypeScript type erasure does not replace tests of invalid runtime inputs.
- **T07:** export `upcomingWorlds(now, count=3)` in both modules. For finite numeric
  `now` and integer `count` from 0 through 8, return a fresh list of exactly count
  `{world, startsAt}` entries beginning at `nextRotationAt(now)` and advancing by
  `ROTATION_MS`. Use the existing world order at each start; exclude the current
  slot. Invalid now/count throws RangeError, including non-number/bool values.
  Check exact boundaries, negative timestamps and cycle crossings; preserve existing
  exports/order tables and client/server parity. No UI wiring or randomness change.
- **T08:** extract and export `boundedDuration(value, fallback, min, max)` and route
  both duration fields through it. Preserve `Math.min(Math.max(Number(value) ||
  fallback, min), max)` including fractional values, defaults, infinities, thrown
  Symbol coercion, mode/session handling and all other exported parser behaviors.
  Test the new CommonJS export directly (absent at baseline). External behavioral
  cases plus an independent source review must confirm the
  helper is actually used for both fields and duplication was removed. Do not use
  filename overlap, patch-text matching or a sole model judge as acceptance.
- **T09:** export `normalizeTagFilter(value: unknown): TagFilter` returning the input
  only when existing `isTagFilter` accepts it, otherwise `all`. Cover every existing
  option, null/missing/arrays/objects, prototype names and case/whitespace variants.
  Preserve preset labels, validators, match behavior and all calendar/write readers;
  no database/RPC changes or normalization of unrelated private-tag fields.
- **T10:** collect only the currently eligible incomplete current/previous/next
  intentions as before. Trim each candidate string, drop empties, deduplicate exact
  trimmed strings case-sensitively, retaining current → previous → next order.
  Preserve phase/completed eligibility and return a fresh array without mutating
  intentions/recap; do not invent behavior for non-string values outside the types.

## Approved preparation settings and budget accounting

Jorge approved these tasks and settings for preparation. Ten tasks reserve up to
50 minutes of dispatch allowance, plus setup/review/cleanup time. These are trial
settings, not tuning evidence or new application defaults:

| Setting | Ordinary task (one turn) | T03 recovery (two reserved phases) |
|---|---:|---:|
| Model-loop iterations | 12 | 6 per phase |
| Tool calls | 24 | 12 per phase |
| Configured dispatch seconds | 300 | 150 per phase |
| Total reported-token allowance, estimate gated | 64,000 | 32,000 per phase |
| Completion-token allowance | 8,192 | 4,096 per phase |
| Per-request output cap | 2,048 | 2,048 |
| Harness / explicitly requested Ollama context | 16,384 | 16,384 |
| Minimum requested output | 256 | 256 |
| Provider request timeout | 60 seconds | 60 seconds |
| Completion recovery | 1 | 1 per phase |
| Identical-call limit | 2 | 2 |

Use compact prompt, full coding tools, thinking off, default temperature zero,
required verification with exact `['python','-m','pytest','-q','test_visible.py']`,
root-instruction loading off, support tools off, and opt-in turn captures on.
Keep memory isolated/unseeded per task. Record actual settings, schemas, adapter
versions and model identity at freeze/execution; model profiles do not configure
all generation options. These are API/runtime settings, not per-task prompt rules.

**Avoid a silent budget reset:** the runtime enforces per-turn limits and has no
persisted cross-turn task allowance. T03 therefore reserves two fixed 150-second,
32,000-total-token/4,096-output/12-tool/6-iteration phases from the start. Keep the
same phase settings before/after restart so profile drift does not invalidate the
fork; unused first-phase allowance is not transferred. The operator may submit
only those two planned phases, never a third full-budget “retry.” All other tasks
allow one model turn. A transport retry keeps the exact key/input and is not a new
turn. These phase-count rules require operator enforcement; no new cross-turn
budget controller is implemented in this preparation.

The nominal allowance sums are not hard latency or tokenizer/billing bounds.
Cancellation can leave in-flight request usage unknown; retain that unknown and
its known prefix, do not report zero or certify token compliance. Record cumulative
known/unknown usage over all phases and total elapsed time including review,
shutdown/restart and cleanup separately. A correctness pass with budget compliance
unknown is not an established within-budget success. Planned recovery and all
operator actions must be recorded even when they are not manual patch repair.

## Client and recovery procedure

- **Ordinary tasks:** start a fresh API session via the assigned client with the
  task's disposable root and captures enabled, submit the frozen request once,
  inspect completion/check/diff evidence, and run external acceptance. Keep all
  statuses, not just accepted artifacts. No operator-supplied model follow-up to rescue a failed task; the configured
  bounded runtime completion recovery remains part of the same turn.
- **T03:** print/retain session/request identity in CLI, submit the frozen calendar
  request once. After the submission key is visible, use a second CLI for one
  explicit `runs --request-id` lookup, then request cancel once if it shows running.
  If no running record is observed or cancellation arrives too late, record a missed cancellation;
  do not rerun until cancellation happens. Wait for owned settlement, restart the
  API with the same settings/storage, and inspect saved run/checkpoint evidence.
  Use CLI reviews and `continue-session` with pinned latest turn and explicit
  uncaptured-state acknowledgment for the one remaining phase. Submit only if
  guards allow; a blocked fork is evidence, not permission to bypass the guard or
  create a replacement standard session. Use only “Continue the same requested
  calendar change, preserving existing work; report actual checks and limits” as
  the frozen continuation input, with no patch hints. Pin its full request/key
  before submission. Record planned interruption separately from unplanned rescue.
- **T05:** grade the final coding artifact first. Then make one recorded later
  change to the separate setup sentinel, not the target; use the CLI's explicit
  single-file reversal for the target. Verify the target equals its captured before
  bytes/mode while the later sentinel and other paths remain unchanged. Re-read
  history after restart without replay. This is a preservation check after scoring,
  not an extra model attempt or a license to reset Git/user changes.
- A run that needs direct HTTP to work around missing client controls fails that
  workflow criterion. Record the blocker and stop that workflow; fix it in a
  separate regression-backed PR rather than improvising a hidden operator path.

## Before execution: unresolved gates and ledger use

1. Task/settings/recovery agreement is recorded in the unrun ledger. It authorizes
   preparation; execution remains gated below.
2. All ten adapters/references/criteria are implemented and validated offline.
   Freeze source/criterion hashes and actual requests/settings in a reviewed
   execution manifest after preflight. Preserve historical definitions; do not
   rewrite old task criteria to fit this trial. Ledger hashes identify preparation
   assets, not a final execution freeze or a future workspace baseline.
3. The [recorded deterministic preflight](CLIENT_PREFLIGHT.md) exercised
   representative real-client/setup paths before inference: commands/path environment, captures and scoped fork eligibility,
   actual CLI/panel controls, and the planned two-phase accounting. Record any
   blocker instead of adding speculative features. Pick the final harness revision
   only after these preparations, then rerun G0 there.
4. Copy the ledger template to a unique report directory outside both source
   checkouts; populate real IDs/hashes/evidence paths. Do not edit or overwrite the
   template into a claimed result. Attempts, completion, acceptance, budget and
   preservation fields remain null until observed; leave manual factual/source
   review pending until performed. Include all ten planned outcomes; no replacement
   tasks, hidden reruns, model switching or post-hoc denominator changes.
5. Report artifact acceptance, completed-and-accepted, observed checks,
   within-budget known successes and preservation/client outcomes separately. The
   roadmap's 8/10 target and broader held-out/two-model release gates remain open.

### Refactor acceptance requires source review

T08's `acceptance/duration_refactor/source-review.md` is an external rubric for
reviewing the final artifact. A correct but unused exported helper can pass the
behavioral suite. The current automated grader's `passed`/exit-zero result covers
behavior and scope only; it cannot establish this structural requirement. Keep
T08 artifact acceptance and completed-and-accepted unknown until the independent
source review confirms reuse for both fields and removal of duplication. Record
the reviewer, verdict, criterion hash and final diff hash outside the workspace.
The reference solution has both fields routed through the helper; that inspection
does not pre-approve any future model patch.

Preparation includes seven new adapters, visible checks, external criteria and
reference patches, plus portable regression fixtures. No provider calls, edits to
the Duodoro working checkout, paid calls, Docker actions or release publication.
