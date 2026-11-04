# Direct prompt feasibility: six responses, no edit, rejected artifact

The single authorized direct-profile attempt reached the six-iteration cap without
editing the ISO-week implementation. Original external acceptance and a separate
copied-workspace regrade both reject the unchanged artifact: **0/1 accepted and
0/1 completed-and-accepted**. The one-cell workflow finished with known usage and
clean unforced settlement. This attempt is closed without retry.

The model listed files, read the implementation and visible test, corrected a
malformed command, ran the failing visible check and emitted a plan. This records
no observed editing or acceptance benefit in this development attempt. It does
not establish general prompt equivalence, a causal effect, held-out quality,
default adoption or release readiness.

## Authority and scope

[PR #358](https://github.com/jorj-pineda/agent-harness/pull/358) merged at
`67198a1f0c7b2bbdf419bc4f57be5900b1d58dac` after independent exact-head reviews
of `3222fcc3704017ddbfb46a68556169ab92e90d9e`, local gates and both CI checks.
Fresh check-only, freeze ancestry and artifact bindings passed before the actual
parent receipt was issued under Jorge's existing request for these three slices.
Private provenance records the verified prerequisites; JSON and Git consistency
alone do not authenticate authority.

The clean retained implementation runtime is
`b23f64e546e7d491a2fe060a2ec2fa978ee9b9c0`. The
[protocol](DIRECT_PROMPT_FEASIBILITY_PROTOCOL.md) and
[freeze](DIRECT_PROMPT_FEASIBILITY_FREEZE_20261104.md) select only fresh ordinal 2:
ho_iso_week / harness / repeat 1 / keep-alive 300 seconds, with the explicit
`direct` profile and `coding-v1-direct-v1` prompt. The other 15 offline capture
fixtures are preservation inputs, never planned live attempts. Prior cohorts
remain closed. Retired-v1 tasks are known development material.

Local Ollama 0.33.2/gemma4:12b used the pinned digest
`4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`.
Original tools, criteria and limits remain: context 16384, thinking off, temperature
zero, 150 dispatch seconds, 32K total/4096 output tokens, 12 tools, 6 iterations,
request cap 2048/min 256. The 300-second value selects keep-alive, not dispatch
budget. No warmup, control, extra attempt, resubmission, manual repair, downloads,
installs, Docker, hosted/paid inference or user-service/configuration changes ran.

An earlier preparation checkout had umask-dependent catalog modes and was
withdrawn before candidate allocation or inference. Its failed logs remain pinned
by the freeze. The replacement verified original bytes and modes without weakening
seals. This environmental preparation failure is not an additional model attempt.

## Recorded outcome

| Planned cell | Ledger | Artifact | Model outcome |
|---|---|---|---|
|2: ho_iso_week / harness / repeat 1|recorded; finished|rejected, acceptance exit 1|iteration_limit; budget_exhausted|

The command's exit 0 means recording and settlement completed, not coding success.
Verification status is `failed`; the completion reason is `Model iteration limit
reached before completion.` The answer is the runtime's iteration-limit placeholder.

| Response | Tool | Observed result |
|---|---|---|
|1|tree|listed the two task files|
|2|read_file|read week_key.py|
|3|read_file|read test_visible.py|
|4|run_command|quoted argv array refused before execution|
|5|run_command|corrected argv ran; original visible check failed|
|6|emit_plan|proposed using the ISO year; no edit followed|

No edit tool was called. The plan is not an implementation or acceptance evidence.
The prior compact feasibility attempt had the same tool-name sequence, but these
separate single attempts are not a matched comparison or evidence of equivalence.
The cap was not raised and the artifact was not repaired. Original-criteria
regrading on a disposable copy returned 1.

## Usage, timing and observations

All six native responses are complete: **13,607 prompt + 171 completion = 13,778
tokens**. There was no cancelled or uncertain submission and no missing aggregate
usage. Six tool calls include the argument refusal and failing visible check.

The command ran from **2026-11-04 16:19:44.734946 UTC** to
**16:21:38.771930 UTC**, an outer 114.036984 seconds. Reported dispatch latency was
**64.351512 seconds**, within the fixed 150-second limit. Setup, grading,
preservation and cleanup lie outside dispatch; this is not a hard end-to-end bound.

| Ollama field | Sum across six complete responses, seconds |
|---|---:|
|total_duration|54.544302667|
|load_duration|12.650034917|
|prompt_eval_duration|27.754092000|
|eval_duration|14.038972000|

Backend component sums are supporting observations, not independent host spans
or a causal speedup comparison. Six post-response owned GET snapshots record
`loaded` for the selected identity and 300-second keep-alive. These points do not
prove continuous residency, prefix-cache reuse, authenticated model identity or
GPU settlement.

## Preservation and closure

The post-run audit verified the clean retained runtime, catalog/criterion seal,
selected runtime/dependency pins, archived capture and exact first wire bytes.
All 15 unused fixtures remain unadmitted and verify. The one-cell ledger has zero
unrun planned cells; unused capture fixtures are not failed attempts.

The original implementation and visible-test bytes/modes are unchanged. Pytest
added five `.pytest_cache` files, retained without reset. Empty result change lists
reflect the existing tracking policy and do not certify literal inventory equality.
The copied-workspace regrade is not another model attempt.

Owned cleanup records settled/passed true, forced false, returncode 0 and no errors.
The audit found the owned process group absent and port 11447 bindable. A subsequent
read-only listener check observed no listeners on 11434 or 11447. These historical,
non-atomic observations provide neither a continuing lease nor process isolation.
No further inference follows. Original no-go, 8-of-10 and G2–G5 remain open.

Retained root:
`/Users/jorge/.local/share/agent-harness/trials/direct-prompt-feasibility-20yw3xng`.
WORK_LOG.md and READINESS_HANDOFF.md retain the closed-run handoff. All 46 indexed
receipt, command, native, observation, backend and audit files matched their hashes.
Frozen source/model/fixture pins remain in the reviewed freeze. This results change
is documentation-only; it does not claim a new full-suite test run.

| Retained evidence | SHA-256 |
|---|---|
| `results-evidence-index.json` | `bd0ceffbc534d886a69a417787287cf9287d4a4d6509b0dc22dd26cd3c52e3d3` |
| `execution-result.json` | `894be7500f2b6944f9f5b3e7e9ecb6a56676ea8d303c0831ec27a2127f5a15cc` |
| `candidate/run/report.json` | `0050eb30995457d9d359c09d2fa0eb4a608f8c0ebcdf795e0669bffe4acdd0e4` |
| `results-audit.json` | `d7d520454c27e8f27f456cb7e8e22f80823e528ef5d8d67b4198d8650d93ec50` |
| `regrade-02.txt` | `0d3f70e83a16c833e8171bcb6f3a03f055a198d7d798efea9817250cf206efaf` |
| `audit-results.py` | `a1928087d0e0ae2731b4df37293283a991ae240015eee3f8bb7129fd6d08279a` |
