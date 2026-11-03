# Residency development comparison proposal

This proposal tests whether keeping the evaluation-owned model resident between
requests changes bounded coding outcomes. It authorizes no inference. It follows
the preserved [timing diagnosis](DEVELOPMENT_TIMING_DIAGNOSIS_20261103.md), not an
assumption that residency fixes prompt evaluation or model reasoning.

## Fixed scope

Use the four retired v1 tasks as known development material, harness mode only,
two repeats for each residency selection: zero and 300 seconds. Sixteen fresh
workspaces are required. This is not held-out evidence. Preserve original task,
fixture, reference and authoritative criterion bytes and registration checks.
Use one clean retained runtime containing the manifest-bound residency support;
the execution freeze must select its exact commit before any generation.

Only the manifest's keep_alive_seconds selection differs between arms. Preserve
the original local gemma4:12b digest and Ollama version from plan.json, compact/full
prompt/tools, explicit Settings and subprocess environment, context16384,
thinking off, temperature zero and all original per-cell limits: 150 dispatch
seconds, 32K total tokens, 4096 completion tokens, twelve tools, six iterations,
2048 request output cap and 256 minimum. No warmup or extra inference.

Schedule tasks in original manifest order. For repeat one, each task runs zero
then 300; repeat two runs 300 then zero. Each cell gets its own owned lifecycle,
fresh workspace and independent image, with empty loaded-model startup observation.
Residency is within a cell only: settle its group before starting another cell.
Do not compare this harness-only experiment to historical minimal rows as a new
prompt-policy comparison. Total dispatch allowance is forty minutes plus setup,
grading, preservation and cleanup; it is not a hard end-to-end deadline.

## Preventive execution requirements

The existing fixed cohort helper does not implement this interleaved schedule:
do not invoke it twice and call the result this protocol. A later focused runner
must enforce this schedule, explicit residency per cell, original grading and
serial owned settlement, retaining all sixteen planned states before dispatch.
Unknown submission, missing usage, provider/transport/identity/runtime/settings/
environment/criteria/preservation failure or failed/forced cleanup stops the whole
remaining experiment, including both arms. Ordinary rejected artifacts or native
budget stops may continue only with known usage, intact pins and clean settlement.
Never retry, reset, manually repair, transfer budgets or substitute a task.

Exact normalized/native first inputs must be captured for every fresh workspace.
Compare paired inputs after only explicitly documented workspace-path normalization;
residency is a backend environment selection, not a prompt change. Freeze both
manifest selections, original criteria, command recipe, executable/dependency/image
pins, report paths, refusal probes and resource checks. Require exact-head
independent review, local gates, both CI checks, merge, actual parent approval and
fresh launch gates before generation. Current user PR-batch authorization is not
an issued experiment receipt. No downloads, installs, hosted/paid calls, Docker,
user Ollama service/configuration management or previous cohort reuse.

## Outcomes and interpretation

Primary evidence is independent final-artifact or structured-answer acceptance,
reported separately with native completed-and-accepted counts per task/repeat/arm.
Keep unrun cells distinct from failures. Report known/unknown aggregate usage,
tool refusals, applied edits, checks, interventions and host dispatch latency.
Preserve native total/load/prompt-evaluation/output-evaluation durations per complete
response; cancelled/missing fields stay unknown, not zero. Native component sums
are supporting prefixes, not independent host timing spans or proof of causality.

Record whether the selected residency is actually observed, rather than assuming
environment configuration proves backend compliance. A nonzero setting does not
prove effective prefix caching. No artifact gain, supported envelope, production
default, held-out improvement or no-go/8-of-10/G2–G5 closure follows automatically.


## Current capture boundary

`capture_inputs(..., prepared=True, residency_schedule=True)` explicitly selects
this schedule and produces a separate `residency-first-inputs-v1` index with the
schedule hash, development label and each cell's residency. Production-adapter
requests still stop at mandatory MockTransport; no generation is forwarded.
Named fixtures are consumed by capture and are not launch workspaces yet.
The legacy verifier/stager deliberately rejects this new format. A matching
residency verifier/stager and executor integration are still required; no existing
manifest or operator can treat these bundles as approved legacy captures. The
ordinary capture default and CLI remain unchanged.


## Residency verification and staging

`verify_residency_capture` accepts only the distinct residency index and verifies
its selected bytes/mode, fixed schedule hash, ordered task/repeat/mode identities
and each residency arm. It retains the existing settings/environment/runtime/
dependency artifact checks, captured file hashes/modes, workspace-root bindings,
prepared/admitted receipts, fixture equality and unknown-inventory refusal.
Legacy `verify_capture` still accepts only the legacy format. Neither entry point
promotes one format into the other or checks actual model residency.

`stage_residency_capture` verifies before allocation, archives consumed capture
bytes in an exclusive disjoint container, verifies them at their original roots,
then creates fresh original fixtures at exactly those paths. It writes the distinct
staged-residency receipt last. Failure preserves partial artifacts without reset;
the operation is bounded/non-atomic and does not isolate external writers. A hash
selects bytes, not approval authority. Runtime provider pin loading, manifest
admission and owned executor integration still need explicit residency adoption;
legacy operators remain unable to launch these bundles.


## Dedicated owned executor

`run_residency_cohort` accepts an explicitly selected residency admission and an
explicit transport factory. Real mode requires retained source provenance. Before
start it validates the selected schedule and ordered admitted cells, captures,
staged inventory and fresh fixtures. Per-request guards recheck source withdrawal,
manifest/plan bytes, capture/runtime/dependency and owned-listener conditions.
Original controls and independent grading use the shared runtime and explicit
command environment. The selected per-cell residency reaches the owned lifecycle.

The executor uses the single-use serial ledger. It writes each result before
provider/backend cleanup; a cell becomes `recorded` only after clean settlement
and final pin checks. A stopped `started` cell can therefore have the last result
row, matched by ordinal order. Later cells remain unrun. Provider cleanup failure
never replaces a primary failure; protocol stop reasons are retained in the ledger.
Reports are replaced through the existing safe report writer. Callers must not
submit requests from transport factories or construct fake admissions as approval.

This is an execution API, not a launch CLI or approval issuer. Synthetic transport
and owned-lifecycle fixtures verify orchestration, not actual model residency or
GPU cleanup. Real residency observation, full actual owned rehearsal/refusal
checks and exact reviewed freeze/receipt/fresh gates remain required before launch.
No default transport, warmup or extra attempt is supplied.


## Loaded-model point observations

Real residency execution wraps each complete provider response with a bounded,
GET-only `/api/ps` observation before returning tool-bearing content. Ownership
and source/pin guards run before and after the GET. Only no loaded model or the
single selected model/digest is accepted. Private response-ordinal records retain
selected keep-alive, timestamp and loaded/not-loaded state. Missing/invalid data,
identity/ownership errors or recording failures latch the provider closed and stop
the experiment. Cancelled requests without complete responses have no inferred
observation; their usage/state stays unknown. No GET retry or warmup is added.

These snapshots do not prove continuous residency, prefix-cache reuse or GPU
settlement. A not-loaded snapshot in the 300-second arm is recorded as observed,
not silently treated as effective residency. Read-only observation time is inside
the existing dispatch budget and can itself fail; no allowance is enlarged. Real
mode performs actual GETs; scripted executor tests omit observations rather than
pretending the canned generation loaded a model. Provider wrapper tests use mock
HTTP metadata and establish mechanics only. Actual owned metadata rehearsal and
exact runtime/input/resource freeze still precede generation.


Provider failures and cancelled generation latch the observation wrapper closed
without becoming an observation failure. Existing inner guard reasons are preserved;
when the cancelled request has no complete usage, normal result checks retain the
missing-usage stop. Only an exception during the post-response observation sets
residency_observation, including cancellation during that observation. This changes
diagnostic attribution, not retry, cleanup or cohort stop policy. Historical frozen
runtimes retain their original labels and are never rewritten.


## Per-cell image ownership

The dedicated executor prepares a new image under each exclusive cell output,
using only the selected immutable owned model bundle as the read-only source.
Existing clone-only preparation independently verifies source and copy, refuses
shared inodes, enforces read-only copied files and writes its receipt last. The
owned backend receives this cell-specific image path, not the shared source path.
Preparation is outside dispatch time, matching the protocol setup boundary.
A clone or post-copy pin-check failure stops both arms before backend/transport or
workspace admission, preserving partial image evidence. No fallback copy method,
download, hardlink, reset or extra inference is added. Tests using synthetic copy
and backend fixtures establish routing/refusal only; actual APFS copy and owned
metadata rehearsal are required on the selected retained runtime before launch.
