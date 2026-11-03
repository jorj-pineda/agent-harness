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
