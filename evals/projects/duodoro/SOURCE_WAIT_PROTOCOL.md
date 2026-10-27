# Source request wait versus dispatch budget: local protocol

**Prepared and unrun.** This is the first of three newly requested PRs: protocol,
independently reviewed execution freeze, then conditional bounded results. The
[fixed plan](source-wait-plan.json) has no live cells, freeze or approval. Protocol
merge alone cannot launch inference or reopen the earlier stopped cohort.

The [retained backend diagnosis](PROJECTED_RECOVERY_BACKEND_DIAGNOSIS.md) shows
prompt and generation progress before the second T02 call's read timeout. Hypothesis:
a 60-second buffered-provider inactivity timeout can end an otherwise active source
before its separately enforced 150-second dispatch allowance. Giving the provider
180 seconds may let that call return, or may merely move the stop to the unchanged
150-second runtime deadline. Neither outcome is assumed. This differs from the
older [T09 comparison](PROVIDER_WAIT_PROTOCOL.md): original T02, full tools and a
150-second source allowance, without any continuation or projected-recovery arm.

## Fixed fresh cells

Use only local Ollama 0.33.2 / `gemma4:12b`, digest
`4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`.
Run exactly R60-1, R180-1, R180-2, R60-2 with provider timeouts 60, 180, 180, 60.
The existing scalar controls connect/read/write/pool inactivity; it is not an
absolute request deadline. Keep buffered transport and the production adapter.

Each cell gets a fresh original T02 full-source archive from Duodoro
`5324cabba4ad3c8ad2e1cd1480218cf087d9b741`, the unchanged original request,
`server/payloadParsers.js` scope, visible adapter and independent acceptance/reference.
Inspect-original-instructions obligations remain. Never replay a captured failed
body, reuse a failed store, edit histories or reset Jorge's checkout. New cells
are not replacements for the three sources/eight arms left unrun previously.

Keep compact/full, unrestricted default reads, thinking off, temperature zero,
16,384 context and explicit Ollama context, estimate ratio 3, request output
2,048/minimum 256, one completion recovery, identical-call limit 2 and file budget 2.
Use checkpoints and the original exact visible check after edits. Root injection,
support and plan gate stay off. Every cell retains **150 dispatch seconds, 32K
total tokens, 4,096 output tokens, 12 tools and six iterations**. Four turns reserve
at most 600 dispatch seconds, 128K total/16,384 output tokens, 48 tools and 24
iterations physically. No budget transfer, extension or extra turn. Setup, snapshot,
storage, grading, shutdown and operator overhead are not a hard end-to-end bound.
CLI response read-inactivity remains 600 seconds.

Use one new literal evaluation-owned workspace root/inode with identical original
bytes/modes per cell, separate empty memory/session stores and private captures.
First normalized messages/schema/allowance/temperature and exact native body must
match across cells; timeout changes only transport settings and captured configuration.
Verify the actual adapter's four timeout phases at 60/180 offline. Later generated
histories may differ; no padding or rewriting to manufacture parity.

Start and stop a fresh owned backend lifecycle per cell using an independently
hashed private APFS CoW image of existing authorized weights. No warmup, model or
dependency downloads, Docker, hosted/paid calls or user-service stop/unload. Freeze
actual owned routes, executable/model/environment/settings and user checkout pins.
Fresh backends clear per-runner history; OS caches, hardware timing and unrecorded
backend state remain uncontrolled. Preserve configuration/identity files; never
expose credentials or `.env` contents. Original user Ollama remains outside ownership.

## Expected outcomes and cohort stops

**This new comparison explicitly treats provider `read_timeout` and native dispatch
wall-budget exhaustion as expected cell outcomes.** They may continue to the next
predeclared cell only after complete settled preservation/grade/regrade and clean
API/backend shutdown. This changes this experiment's stop policy; it does not alter
the previous cohort, product runtime or original source/arm rows.

Classify after inspecting the finalized native run. A returned read-timeout failure
must have the provider's exact retained `read_timeout` kind. A wall-budget outcome
must have native `budget_exhausted` status and the configured wall-budget completion
reason. An observer seeing `CancelledError` alone cannot grant continuation. Retain
unknown in-flight usage; no expected failure becomes zero usage or budget success.
Other normal bounded token/context/tool/iteration stops and completed turns also
remain outcomes. Do not force a timeout, cancellation, second call or target edit.

Stop all remaining cells for any other provider/transport failure, returned missing/
malformed usage, unexplained cancellation, input/recording/storage failure, changed
model/runtime/environment/criteria, uncertain submission without a finalized original-
key record, external mutation or unsettled/incomplete preservation. No replacement,
retry, fallback or manual repair. Unknown delivery permits original-key GET only;
never another POST or new key. Every failed/blocked/unrun row remains visible.

Before any next-cell staging, require complete eligible bytes/modes/type/directory
seals, independent grades/regrades, exact current agreement with the last scored
artifact and unchanged protected source/criteria/reference evidence. Preserve the
owned root/inode and refuse unknown work. Both owned endpoint lifecycles must be
settled; timing out an await is not proof of process exit. Keep raw finalized run,
archive, checkpoints, closed stores, paired inputs and backend/progress logs. Perform
GET-only restart review on an independent closed-store clone, preserving the actual
run/store bytes and refusing provider dispatch. No session fork or recovery run.
Staging remains bounded/non-atomic; path rules and process groups are not isolation.

## Required freeze and reporting

Before inference: fresh original initial-fail/reference-pass proofs; actual CLI/local
HTTP scripted source and GET-only restart for both timeouts; explicit read-timeout
and native wall-cancellation expected-outcome cases; returned-missing-usage/other-error/
unknown-key/preservation refusal cases; exact first input/native-timeout guards;
managed process cleanup; absent scheduled state; exact-runtime focused/full/static G0
and both labeled scripted matrices. Pin all recipes/evidence/model/environment/input
contracts in a new public freeze, independently review its exact head with Luna,
pass both CI checks, merge, record parent approval and recheck fresh per-cell gates.
An incomplete gate leaves cells unrun. No script or historical approval is a lease.

Primary result: completed-and-independently-accepted artifacts per arm /2. Also
report acceptance alone, observed checks, second-call terminal response versus
not-reached/failed, provider-timeout versus dispatch-budget boundaries, every change,
known/unknown usage and time scopes. Do not rank a cohort stopped incomplete. Backend
progress counters never replace terminal provider usage or acceptance. Unknown
aggregate usage or any reported phase overshoot cannot certify budget success.

Two low-variation repeats of one public task cannot establish held-out quality,
a useful unattended envelope, a default timeout promotion or a projected-recovery
benefit. No-go, original cohorts, 8-of-10 and broad G2–G5 remain unchanged even if
180 returns a response. The original request/criteria and fresh fixture's initial
visible/external failure and unchanged reference success validate preparation only.
Private receipt: the plan's `preparation` path/hash. No inference ran in this slice.
