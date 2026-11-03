# 300-second residency feasibility: loaded model, rejected unchanged artifact

The one authorized fresh attempt finished its six allowed model iterations with
known usage and clean settlement. All six post-response snapshots show the model
loaded under the 300-second setting. The model read the task, corrected a malformed
command, ran a failing visible check and emitted a plan, but applied no edit.
Original external acceptance rejects the unchanged artifact: **0/1 accepted and
0/1 completed-and-accepted**. The planned feasibility run is closed without retry.

The observed stopping boundary is the six-iteration cap. Dispatch lasted 63.912269
seconds within the fixed 150-second limit. This one development attempt establishes
that the selected configuration returned all six responses while the model was
observed loaded; it establishes no causal residency speedup, coding-quality gain,
held-out result, default adoption or release-readiness closure.

## Authority, runtime and scope

[PR355](https://github.com/jorj-pineda/agent-harness/pull/355) merged at
`f757844659905673426116daa9d3157ffde09945` after exact-head review of
`9e61aac6f88ea45ee8622d0cde20500f3c7dd490`, local gates, independent verification
of 30 selected hashes and both CI checks. Fresh candidate preflight, freeze ancestry
and recipe/manifest/G0 bindings passed. The actual parent issued its selected
receipt under Jorge's request for these three slices; private provenance records
the authorization and verified prerequisites. JSON/Git consistency alone does not
authenticate authority.

The clean retained implementation runtime is
`3e5caa9cdc60fd5fd6c6c9a955bfff960725656c`, containing the merged #354 stop-cause
fix and reviewed one-cell support. The [freeze](RESIDENCY_300_FEASIBILITY_FREEZE_20261103.md)
and [protocol](RESIDENCY_300_FEASIBILITY_PROTOCOL.md) select only original residency
ordinal 2: ho_iso_week, harness, repeat 1, keep-alive 300 seconds. The other 15 offline
capture fixtures are verification inputs and never planned live attempts. The old
stopped sixteen-cell comparison remains closed. These retired-v1 tasks are known
development material, not held-out evidence or a matched control comparison.

Use of local Ollama 0.33.2/gemma4:12b retained the original model digest, prompt/tools,
context 16384, thinking off, temperature zero and all limits: 150 dispatch seconds,
32K total/4096 completion tokens, 12 tools, 6 iterations, request cap 2048/min 256.
No warmup, control attempt, resubmission, manual artifact repair, budget transfer,
downloads, dependency/model installs, Docker, hosted/paid calls or user-service/
configuration management occurred.

## Recorded outcome

| Planned cell | Residency | Ledger | Artifact | Model outcome |
|---|---:|---|---|---|
|2: ho_iso_week / harness / repeat 1|300 seconds|recorded; ledger finished|rejected, exit 1|iteration_limit; budget_exhausted|

The process returned 0 because the one-cell workflow recorded its result and
settled normally. It does not mean the coding task passed. Completion reason:
`Model iteration limit reached before completion.` Verification status is `failed`.
The result answer is the runtime's iteration-limit placeholder, not a completion
claim by the model. No edit tool was called, no source change was applied, and no
manual patch rescue occurred.

Six complete native responses produced exactly these six tool calls:

| Response | Tool | Observed result |
|---|---|---|
|1|tree|listed the two task files|
|2|read_file|read week_key.py|
|3|read_file|read test_visible.py|
|4|run_command|quoted argv array refused before execution|
|5|run_command|corrected argv executed; original visible check failed|
|6|emit_plan|planned to use isocalendar's ISO year; no edit followed|

The final plan identifies the intended change but is not an implementation or
acceptance evidence. The fixed cap prevented another model iteration. We did not
raise that cap, repair the artifact or add an attempt. An isolated copied-workspace
regrade against the original external criteria returns 1, reproducing rejection.

## Usage, timing and residency observations

All six request/response bodies are complete. There is no cancelled or uncertain
submission and no missing aggregate usage. Reported prompt tokens 13,559 plus
completion tokens 171 total 13,730. Six tool calls include one argument-validation
refusal and one correctly executed visible check that failed on the artifact.

The command ran **2026-11-03 21:58:07.541192 UTC** to
**21:59:58.426676 UTC**, an outer 110.885484 seconds. Reported dispatch latency is
63.912269 seconds. Setup, grading, preservation and cleanup lie outside dispatch
time; these spans are not a hard end-to-end deadline guarantee.

| Ollama field | Sum across all six complete responses, seconds |
|---|---:|
|total_duration|53.857377834|
|load_duration|11.100977459|
|prompt_eval_duration|28.518879000|
|eval_duration|14.126533000|

These backend component sums are supporting timing observations, not independent
host spans or a causal comparison. All six post-response owned GET snapshots show
`loaded` for selection 300 and the selected model identity; startup metadata had no
loaded model. Snapshots do not prove continuous residency, effective prefix caching,
model attestation or GPU settlement. The previous zero-second attempt's cancelled
tail and different run conditions cannot supply a matched speedup denominator.

## Preservation and closure

The read-only post-run audit verifies the clean runtime, original catalog/criterion
seal, selected runtime/dependency pins and archived capture. The submitted first
wire bytes match 02-wire.json exactly. All fifteen unused captured fixtures verify
and remain unadmitted; no other cell output was created. There are no unrun planned
cells in this one-cell ledger. Unused capture inputs are not failed attempts.

Original week_key.py and visible-test bytes/modes are unchanged. Final literal
inventory equality is false: pytest generated five files under `.pytest_cache`.
Those files are retained without reset. Empty added/modified/deleted lists follow
the existing change-tracking policy and do not certify literal inventory equality.
The regrade used a separate disposable copy and was not another model attempt.

Owned cleanup records settled/passed true, forced false, returncode 0, no errors.
The post-run audit found its process group absent and owned port 11447 bindable;
no listeners were observed on 11434 or 11447 after exit. These are bounded historical
observations, not a continuing lease or process isolation. The original no-go,
8-of-10 target and G2–G5 remain open. No further inference follows from this result.

Retained root:
`/Users/jorge/.local/share/agent-harness/trials/residency-300-feasibility-6wx3aale`.
WORK_LOG.md records preparation, gates, approval, command and outcome. The result
index pins 46 command/receipt/native/observation/backend/audit files; frozen source,
model and original fixture bytes remain selected by the prior freeze/seals.

| Retained evidence | SHA-256 |
|---|---|
| `results-evidence-index.json` | `ac179b1c05014eb472ad7f23c229991b5c7d23d974df85264b3faf8bc5cb0416` |
| `execution-result.json` | `7606d7eafeb7f05a9d8823e1e989fdffb6816265c16bd56534075ff747fd80d3` |
| `candidate/run/report.json` | `5efaa7f345516835fee954eee7003477d2eb849bd2f6d213da144b544e1aa2d3` |
| `results-audit.json` | `6a100a59ce8b61cc44d8ad849e531f256bc842d91c497fedcf61b85a44dac224` |
| `regrade-02.txt` | `8b96de60e54c0ec3482cd8142a9ed3c5813f87b2c698a30d998dd80f13494815` |
| `audit-results.py` | `f14ead50a24bfbabc7121e80fd661aa7432d87ddc67c57815b2358026e7c4a85` |
