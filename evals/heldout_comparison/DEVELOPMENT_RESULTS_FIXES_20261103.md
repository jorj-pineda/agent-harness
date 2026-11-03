# Gemma fixes development validation — stopped

The fresh authorized development run stopped after its first cell on 2026-11-03
UTC. Gemma corrected a quoted command-array string after compact validation
feedback and successfully invoked the configured visible pytest command. That
check failed on the unchanged bug. No source edit followed before the 150-second
dispatch limit cancelled the sixth submitted request. Aggregate usage is unknown,
so the protocol stopped the remaining fifteen cells. **No accepted patch or
completed comparison was observed.** This cohort is closed without retries.

This is development on retired v1 tasks, not held-out evidence. Fresh workspaces
do not undo author/reviewer/tuning exposure. The old #335 cohort remains closed
and untouched. Its lack of minimal results also prevents a matched causal estimate
of #336–338. One corrected command is an observation, not an attributed gain or
proof of dependable recovery. No original no-go, 8/10 or G2–G5 closure follows.

## Authorization and execution

Jorge explicitly requested “test the fixes on fresh dev!” The parent recorded
that grant, verified both independent exact-head source/artifact reviews and CI,
merged [freeze #339](https://github.com/jorj-pineda/agent-harness/pull/339), passed
fresh check-only, and issued the selected actual parent receipt. Receipt fields
are assertions backed by that context, not independent authentication.

- Runtime: `cf52e2e723a93c570947bdb114d22dd2799d03b5`, including #336–338.
- Reviewed freeze head: `f938a5c35873ebe7b82be898c9d45a96a5eac2ef`.
- Freeze merge: `63ee6e7e15e3dba8f7b4351284210c6508f35bb5`.
- Manifest SHA-256: `8600423984f80fbd14ecaf74961a33fb51f409df24fdb67ebfb53c5b1631ef02`.
- Local Ollama 0.33.2 / `gemma4:12b`, digest
  `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`.
- Invocation 01:21:04.886736–01:23:53.252368 UTC, child exit 1
  (`ProtocolStop: missing_usage`). Wrapper exit 0 means logging completed, not
  cohort success. Cell dispatch latency: 150,005.12 ms.

The [development freeze](DEVELOPMENT_FREEZE_FIXES_20261102.md) retains original
order, criteria, settings and per-cell budgets: 150 dispatch seconds, 32K total /
4096 output tokens, twelve tools, six iterations. Setup/grading/storage/cleanup
remain outside dispatch time. No retries, warmup, installs, downloads, Docker,
hosted or paid calls, task substitutions, manual patch repairs or user-service
management occurred. The external existing interpreter and explicit retained
source/environment were used. Neither Ollama port had a listener before launch.

Retained runtime G0 passed 2,825 offline tests; proposal head passed 2,836 (both
five live deselected, one warning). Static gates, both scripted matrices, sixteen
mock workflows and thirteen refusal probes passed. These are mechanics evidence.

## Every planned cell

| Cell | Task | Mode | Repeat | Result |
|---|---|---|---|---|
| 1 | ho_iso_week | harness | 1 | rejected; wall limit; usage unknown |
| 2 | ho_iso_week | minimal | 1 | unrun — missing_usage |
| 3 | ho_rolling_mean | minimal | 1 | unrun — missing_usage |
| 4 | ho_rolling_mean | harness | 1 | unrun — missing_usage |
| 5 | ho_endpoint_record | harness | 1 | unrun — missing_usage |
| 6 | ho_endpoint_record | minimal | 1 | unrun — missing_usage |
| 7 | ho_cache_policy | minimal | 1 | unrun — missing_usage |
| 8 | ho_cache_policy | harness | 1 | unrun — missing_usage |
| 9 | ho_iso_week | minimal | 2 | unrun — missing_usage |
| 10 | ho_iso_week | harness | 2 | unrun — missing_usage |
| 11 | ho_rolling_mean | harness | 2 | unrun — missing_usage |
| 12 | ho_rolling_mean | minimal | 2 | unrun — missing_usage |
| 13 | ho_endpoint_record | minimal | 2 | unrun — missing_usage |
| 14 | ho_endpoint_record | harness | 2 | unrun — missing_usage |
| 15 | ho_cache_policy | harness | 2 | unrun — missing_usage |
| 16 | ho_cache_policy | minimal | 2 | unrun — missing_usage |

Observed artifact acceptance: 0/1; native completed-and-accepted: 0/1. Harness has
one rejected coding artifact and seven unrun cells; minimal has eight unrun cells.
No repository question ran. Unrun cells are not observed failures or a 0/16 score.

## Native sequence and limits of the observation

1. Tree listing, source read, and visible-test read returned normally.
2. The fourth response supplied `run_command.argv` as a quoted JSON string.
   Registry validation refused before dispatch with the new compact field/message
   format and the existing separate-elements correction example (#337 active).
3. The fifth response supplied the correct five-element array. The configured
   `python -m pytest -q test_visible.py` ran and exited 1: one failing test.
   The original incorrect ISO year behavior remained; verification is `failed`.
4. The sixth request was delegated but cancelled at the wall limit. No complete
   response or usage was retained. Native row termination is `budget_exhausted`;
   cohort stop reason is separately `missing_usage`.

#336's unknown-executable guidance was not exercised; #338's exhausted identical
failed-batch gate was not exercised. Their benefits cannot be inferred from this
run. #337's compact error was delivered and a corrected call followed, but the
same model/task alone does not isolate cause or establish repeatability.

| Complete native response | Prompt tokens | Output tokens |
|---|---:|---:|
| 1 | 1,893 | 7 |
| 2 | 1,968 | 23 |
| 3 | 2,139 | 23 |
| 4 | 2,334 | 37 |
| 5 | 2,443 | 41 |

The known prefix is **10,777 prompt / 131 output tokens**, not a workflow total
or proof of token-budget compliance. Cancelled-call usage and eventual response
remain unknown. Do not resubmit that request or use later cells as retry slots.

## Preservation, regrading and cleanup

The original source and visible-test bytes/modes match the fixture; the report's
diff is empty. The visible pytest invocation created `.pytest_cache`, retained
in place. Full final inventory equality is therefore **false**. The first audit's
strict fixture verifier correctly refused the extra entry; its script is retained
as `audit-results-initial-refused.py`. The corrected audit records the cache and
checks original file bytes/modes explicitly. This is a final-audit scope correction,
not a runtime preservation failure or permission to reset the workspace.

Original authoritative acceptance rejected the artifact (five failed, three
passed); isolated disposable-copy regrading reproduced rejection. Catalog,
criterion identity, selected runtime/dependency pins and exact first wire input
verify; all fifteen unrun fixtures remain fresh. These bounded observations are
non-atomic and do not establish dependency closure or model authentication.

Owned cleanup records settled/pass true, forced false, return code 0, no errors.
The later audit found the owned group absent and its port bindable; listener
inspection found neither Ollama port listening. Startup metadata matched selected
version/digest with no loaded model. Its GET-only `generation_requests: 0` and
`inference_authorized: false` describe startup observation, not the later six actual
submitted requests. The backend failure record propagates ProtocolStop; no success
session receipt is asserted. Historical settlement is not a continuing lease or
proof of absence of arbitrary detached work. The user's service was never managed.

## Retained evidence

Root: `/Users/jorge/.local/share/agent-harness/trials/gemma-fixes-dev-15yxnrks`.
`EXECUTION_LOG.md` retains approval, exact command, gates, stop and audit correction.
Raw report, all sixteen ledger states, tool results, native bytes, process and
cleanup records remain there. The evidence index hashes selected immutable result
files; it binds bytes, not authority or power-loss durability.

| Relative evidence | SHA-256 |
|---|---|
| `parent-approval.json` | `1f113d90038e2aed54ad371e5530f7864de9a23105437014d70a0c25b5ff232a` |
| `execution-result.json` | `22d5bd5dc27566911f752e736a13aa5f396297d974a8235f02006358e33608aa` |
| `execution.log` | `059d64d290d4637d22be64e8cf1be5e8a192bb3b8627279ed12e6142b037e847` |
| `candidate/run/report.json` | `c0e02a6af4e52e980202c9d0c8d2e4011662de00335c0eb034abf1d43b3f921a` |
| `candidate/run/cell-01/backend/cleanup.json` | `1a497626d012d30ebe088ac9c3f2604b6eb014164cca1ae89e2c8419152051cd` |
| `candidate/run/cell-01/native/request-06.failure.json` | `6f7d907b2fd9693ec780e001c79631bd1ba0f82f45127b691e084a1af53098a0` |
| `results-audit.json` | `d1d159f37c21f24cbaed6042c1bd8ed0c39a171e3339a6b52c554719421b0ed8` |
| `results-evidence-index.json` | `0eb0b7981a30ba9b5497a2e3b10da27f40a927aa3d777f8158243ee9d1e1db3d` |
| `regrade-output.txt` | `aafb703935821df46d41357f7b88bb0e8b48259fd2f44fc15ff2c9dc5c2c305f` |
| `audit-results.py` | `76ce7917315fb2725d1467503c7db62b5d92a1883134097483b38dfce11f8433` |
