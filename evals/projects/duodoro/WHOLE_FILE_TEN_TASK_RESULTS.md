# Whole-file ten-task client cohort — 2026-10-22 UTC

**2/10 accepted artifacts; 2/10 completed-and-accepted; 2/10 acceptable with known
reported usage within the nominal budgets. No-go for this unattended task envelope.**
The eight-of-ten target was missed. Keep feature batches frozen; this result does
not justify changing the default preset, raising limits or declaring release readiness.

All ten original tasks ran once through their assigned actual clients, in order,
following the reviewed [protocol](WHOLE_FILE_TEN_TASK_PROTOCOL.md) and
[execution freeze](WHOLE_FILE_EXECUTION_FREEZE.md). PR #173 merged at `010c5ef`
before inference. The execution runtime remained clean published
`d1e5778c834da2165da47cb877dc0d0ced026061`; this reporting branch was not the runtime.
Duodoro remained pinned at `5324cabba4ad3c8ad2e1cd1480218cf087d9b741` in disposable
full-source copies. No Duodoro working-checkout, index or branch edits occurred.

[Structured observations and local evidence hashes](../../results/2026-10-22-duodoro-whole-file-ten-task.json)
retain every outcome, run/request identity, phase, criterion/diff hash, usage prefix,
audit and operator correction. The prepared freeze/plan and original trial files
remain unchanged. The original full-preset cohort remains **2/10 accepted artifacts,
1/10 completed-and-accepted**; these counts are a separate whole-file cohort.

## Task outcomes

“Passed” below is the configured visible check, not independent acceptance.
Unknown completion/checks on failed runs mean no final response was retained.
Usage totals include prompt and output; unknown is never zero.

| Task / client | Independent artifact | Completion / observed check | Reported total tokens | Outcome |
|---|---|---|---:|---|
| T01 formatter / CLI | Accepted | completed / passed | 59,939 | Both formatters handle negative/non-finite values; protected behavior/files pass. |
| T02 timer normalization / panel | Rejected | unknown / unknown | unknown | Provider read timeout after inspection/plan, no source change. |
| T03 calendar / CLI | Rejected | cancelled, then budget_exhausted / not_run | unknown | Original early cancellation; restarted fork reaches its 150-second dispatch limit without an edit. |
| T04 share storage / panel | Accepted | completed / passed | 29,974 | Storage failures/accessor exceptions and preserved token/URL behavior pass. |
| T05 timer preferences / CLI | Rejected | incomplete / not_run | unknown | Missing output-token usage stops the turn; no source change. |
| T06 shortcut count / panel | Rejected | completed / passed | 29,014 | Uses `Number.isInteger`, accepts unsafe integers; external edge cases fail. |
| T07 rotation preview / CLI | Rejected | unknown / unknown | unknown | Provider read timeout, neither module changed. |
| T08 duration refactor / panel | Rejected | unknown / unknown | unknown | Provider read timeout, helper absent; executable acceptance fails. |
| T09 tag filter / CLI | Rejected | unknown / unknown | unknown | Provider read timeout, export absent. |
| T10 wrap-up intentions / panel | Rejected | unknown / unknown | unknown | Provider read timeout, trimming/deduplication unchanged. |

Six bugfixes yield 2/6 accepted and completed-and-accepted; three features yield
0/3; the refactor yields 0/1. All ten final grades reproduced on retained scored
copies. Every protected-file/mode check passed. Only T01, T04 and T06 changed a
source file; no tests or unrelated eligible files changed. T08 source review was
not performed or counted as passing: failed executable acceptance already rejects
the artifact. Its unchanged source has no exported helper to review as a success.

T06's final answer explicitly claims positive safe-integer handling, while its
patch only tests integer-ness. That claim is false despite its real visible-check
pass. Limited source/tool review of T01/T04 supports their formatter/storage and
check statements; no complete semantic answer score or model judge is claimed.
Their comment-only Python illustrations are answer text, not executed Python
coverage; the actual visible adapters invoke Node.

## Frozen limits and usage

Local `gemma4:12b`, Ollama 0.33.2, digest
`4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c` used
compact/whole-file, temperature zero, thinking off, explicit 16,384 context and
2,048 output tokens per request. Nine ordinary tasks reserved 300 dispatch seconds,
64,000 total reported / 8,192 output tokens, 24 tools and 12 iterations each.
T03 used two identical reserved halves: 150 seconds, 32,000 / 4,096 tokens,
12 tools and six iterations per phase. Eleven coding turns total, no transfer,
third phase, replacement attempt, prompt hint or manual patch repair.

All recorded first-source request identities and configured profiles match the
freeze. The fork input depends on actual retained history and native guards.
T01/T04/T06 have known reported totals within their nominal token/output/tool/
iteration allowances. Only T01/T04 are also accepted. Seven task totals remain
unknown, including T03's cancelled/in-flight phases and T05's incomplete usage.
The journals retain known response prefixes separately; even a zero known prefix
does not establish zero consumption or within-budget success. Missing final-response
tool records on provider failures are unknown, not zero actual calls.

Five tasks failed with durable `read_timeout` metadata. This identifies the
provider boundary observation, not a proven underlying model/server cause or a
guaranteed output-size diagnosis. No timeout increase or provider retry was added.
The CLI's frozen `--response-timeout 600` is HTTP inactivity tolerance, not an
extension of server dispatch limits. Loop latency, journal lifetime and process
lifetime are distinct; setup/final review/storage/cleanup can exceed dispatch time.
The aggregate 3,000-second reservation is not total wall duration or a hard return
deadline. Provider usage reports do not attest backend compliance.

## Client and preservation observations

Five tasks used actual CLI `chat`; five used rendered panel New session, captures,
provider/workspace and Send controls with native SSE. Each retained its exact
request key. HTTP journals show one original coding submission per task plus
T03's one guarded continuation. No panel fallback, injected app/API submission,
new-key retry or replacement browser identity was used.

The panel displayed completed/check/diff evidence for T04/T06 and saved failures
for T02/T08/T10. Original-key Check run retrieved those failure records and cleared
pending keys. Selecting T04 saved history exposed unchanged workspace, valid
transcript and matching configuration; T06 history/workspace/transcript reviews
were also accessible with composer disabled. Restarted CLI original-key/run and
workspace/transcript/configuration reviews retrieved unchanged T01/T07/T09 evidence.
T05's saved run was equal after its review-only restart. These are the exercised
paths; this cohort does not re-establish every panel control after a server restart.

T03 used the original early-running-key trigger, not the formatter probe's
wait-for-write trigger. One CLI cancellation settled without an edit. The API
stopped, model unloaded, and the same root/settings/storage restarted. CLI reviews
found matching configuration/valid transcript, pinned the actual latest turn,
and one acknowledged continuation was admitted with the frozen key/message.
It exhausted the reserved dispatch time without an edit/check. The source run
was equal afterward. This exercises controls but does not demonstrate recovery
of a partial patch or acceptable calendar work.

T05 was copied and graded before the exact later sentinel line was appended.
Its target remained equal to the frozen before state, so no eligible reversal
existed and none was submitted. The later line survived restart; the full eligible
manifest/modes match baseline except that declared line. This is preservation
evidence, not a successful live reversal. No reset or bypass substituted for it.

## Operator corrections and observation limits

T3 snapshots repeatedly returned “Preview automation snapshot timed out after
15000ms.” Native computer-use fallback also returned “Sky Computer Use request
timed out” before reaching the tab. The existing T3 tab retained its generated
owner, profile and origin; its normal typing/select/click controls and read-only
DOM inspection worked. Those controls submitted the frozen panel tasks. No app
functions, fetch, EventSource or localStorage writes were injected. Screenshot/
semantic-snapshot coverage is unavailable for this cohort; DOM/state observations
and native SSE/client/server journals are retained. No task was submitted during
the failed observation attempts.

Local reporting helpers were corrected before final reporting: `token_usage`
replaced an incorrect `usage` field lookup; missing response tool records remain
null; T08's executable failure is rejected without asserting source-review success;
the ledger explicitly maps acceptance. A stale cleanup message was corrected to
match the lifecycle count. These are retained reporting/operator corrections, not
runtime changes, task-code repairs, inference retries or new trial attempts.

## Validation, cleanup and retained evidence

The final audit reproduced all ten external exit codes and checked scored/live
eligible manifests/modes, frozen criteria/reference/settings/recipes/binaries/
packages/root identities/runtime, captured profiles/initial identities, source
run equality, unchanged saved runs/sessions/reversal journals and submission counts.
T05's declared sentinel is the only later eligible-file difference. The scheduled
API served all four frozen UI asset hashes. Fifteen owned APIs exited: eleven
model phases plus four review-only restarts. Port 18755 is closed; evaluation-owned
Gemma is unloaded; the pre-existing Ollama service is preserved; the retained tab
was closed. No dependency installation, paid/new-model call or Docker action.

Raw full scored workspaces, bounded captures, diffs, transcripts, run/session stores,
budget journals, client/server/HTTP logs, grades/regrades, lifecycle and recipes
remain locally at:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-whole-file-live-elep2slp`

The structured report hashes scoped evidence and stores; it is not a downloadable
release bundle. Frozen preparation passed 82 focused / 1,363 offline tests (five
live deselected), Ruff/format/mypy and both scripted 90/90 matrices at the execution
runtime. These historical mechanics checks were not rerun as model evidence for
this docs/data-only report. Fresh report validation checks JSON counts/hashes and
reproduced artifact grades; no application/evaluator behavior changed.

## Decision and remaining work

G3 is missed again; retain no-go for the frozen unattended ten-task envelope.
Broader G2 stays open: this cohort did not recover a partial edit or apply an
eligible reversal. Earlier selected CLI/panel recovery observations retain their
own request/usage limits. G4 held-out/repeated/two-model work and Jorge's G5 release
decision remain unmet. No milestone is silently completed or release approved.

One observation per public development task, different absolute paths and the
different original runtime/preset/client wait prevent a causal comparison or
general quality claim. Do not pool these counts with selected formatter successes
or overwrite the original cohort. The next decision should be an explicit narrower
supported envelope or a bounded offline investigation of the retained failure
records. No automatic feature batch, new model, longer timeout or larger budget
is authorized by this miss.
