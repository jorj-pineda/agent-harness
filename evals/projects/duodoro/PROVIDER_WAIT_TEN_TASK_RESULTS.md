# Ten-task provider-timeout follow-up results

**2/10 accepted artifacts, 2/10 completed-and-accepted, and 2/10 accepted with
known reported usage within the nominal budgets. Retain no-go for the agreed
unattended envelope; the unchanged 8/10 target was missed.** All ten tasks ran
once through their assigned clients, with T03's one reserved continuation: eleven
coding runs total. No manual patch repair, replacement task or additional phase.

The reviewed [protocol](PROVIDER_WAIT_TEN_TASK_PROTOCOL.md) and
[execution freeze](PROVIDER_WAIT_TEN_EXECUTION_FREEZE.md) remain unchanged. PR #196
merged before inference. Execution used runtime
`e2a0d46c3c014cf4dcaea33889929926e47875a7`, original Duodoro `5324cab`, local
`gemma4:12b` and provider timeout 180 within unchanged 300/150-second dispatch
allowances. [Structured results](../../results/2026-10-24-duodoro-provider-wait-ten-task.json)
retain every outcome, grades/regrades, source review, input/budget/progress
metadata, saved-review and cleanup evidence. Observation completed on October 24
UTC, October 23 in Jorge's timezone.

## All ten outcomes

| Task / client | Final coding outcome | Observed check | Independent acceptance | Recorded failure or limit |
|---|---|---|---|---|
| T01 / CLI | completed | passed | accepted | none recorded |
| T02 / panel | dispatch budget exhausted | not run | rejected | 300-second stop; unrelated parser whitespace regression |
| T03 / CLI | source cancelled; fork budget exhausted | not run | rejected | original early cancel before edit; fork's 150-second stop, no patch |
| T04 / panel | completed | passed | accepted | none recorded |
| T05 / CLI | incomplete | not run | rejected | truncated response; no edit |
| T06 / panel | incomplete | failed | rejected | invalid TypeScript artifact; later response truncated |
| T07 / CLI | incomplete | not run | rejected | missing reported output usage; no edit |
| T08 / panel | completed | passed | rejected | independent source/behavior review found an unrelated parser regression |
| T09 / CLI | dispatch budget exhausted | failed | rejected | 300-second stop after an incorrect edit |
| T10 / panel | incomplete | not run | rejected | truncated response; no edit |

T01 reported 59,131 prompt / 969 output tokens and loop latency 180,743.296 ms.
T04 reported 29,447 / 1,145 and 151,603.334 ms. Both completed within their configured
request/tool/iteration and reported token allowances. Four task totals remain
unknown: **T02, T03, T07 and T09**. Known contiguous response-prefix usage is kept
separate; interrupted in-flight requests and missing fields are not assigned zero.
The other rejected tasks can have known reported totals without becoming successes.
Counts are reported usage, not billing or proof of backend resource compliance.

There were 62 recorded requests and 58 returned responses across eleven runs.
No saved run reports a provider `read_timeout`; outer dispatch stops, truncation,
missing usage and incorrect artifacts remain. This is not proof that backend
problems are solved. Latencies measure the saved loop scope; setup, final workspace
review, storage, client observation and cleanup have separate scopes and can add
time beyond the dispatch allowance.

## Acceptance and workflow details

T02 changed an unrelated intention parser's whitespace normalization. T08 extracted
and exported `boundedDuration`, routed both duration fields through it and removed
the duplicate inline expressions, but changed the same unrelated whitespace regex
from `/\s+/g` to `/\\s+/g`. Independent acceptance rejected the regression; the
frozen source-review rubric also received a recorded **fail**, with reviewer,
criterion and diff hashes. Its passing visible helper check and `completed` state
are not semantic certification. No reference-patch matching or model answer was
used to accept an artifact.

T06's visible check failed and external Node execution rejected invalid TypeScript.
The subsequent truncated response is a separate observation, not an established
cause of the earlier invalid edit. T09's failed check and rejected artifact remain
separate from the earlier selected T09 comparison, which accepted 2/2 at 180
seconds. That selected result did not predict this new attempt.

T03 used the original running-key lookup/cancel trigger before an edit, settled and
restarted with the same half-budget settings/storage, passed native continuation
guards and used exactly the frozen continuation input/key. The fork made no patch
before its phase limit. Both saved records remained identical after restart.
Interrupted usage remains unknown; this does not establish partial-edit recovery.

T05 was fully archived and graded before the exact later sentinel append. Its target
was unchanged; the one native reversal request returned HTTP 409 and restart GET
returned `null`, with no action record or applied reversal. The target already
matched frozen before bytes/mode; the later sentinel and every other full-tree
entry were preserved. This preservation observation is not a successful eligible
reversal. No retry, bypass or reset followed the refusal.

## Audit, preservation and cleanup

Every saved source/fork record matches SQLite and its native CLI GET restart export.
All five panel tasks used actual native session/composer/SSE controls and saved
history after restart. Exactly ten source submissions and one guarded fork are
retained; GET-only review generated no model call. Planned cancellation and the
sentinel append/reversal attempt are recorded separately from coding rescues.

All 62 normalized/wire pairs match saved request identities and production adapter
serialization reconstructed **offline with MockTransport**. Tool-reply continuity,
private file hashes and 0700/0600 permissions match. No raw history or wire payload
is published. Input presence is not backend receipt, generation progress or chunk
timing; recording I/O contributes to measured adapter waiting.

All full artifact trees, bounded diffs, external grades/regrades, protected paths
and modes match the retained evidence. The independent audit verified 360 outcome-
evidence hashes plus the immutable freeze's source/recipe/preflight hashes. Original
criteria, setup, references and historical cohorts remain unchanged. Duodoro was
never the model-editing workspace; recorded head/status/index/refs fingerprints
match before and after. No Duodoro commit/push, paid/new model, dependency install,
Docker action or Ollama restart was performed.

All **22 owned APIs** stopped with complete shutdown receipts, port 18761 is released,
Gemma is unloaded and the owned browser tab is closed. The existing Ollama service
remains. Local raw artifacts, plaintext inputs and receipts remain under:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-provider-wait-ten-live-t152x3td`

A preliminary read-only reporting query expected a reversal object but received
`null`. The final audit retains the null and HTTP refusal. No execution recipe,
model input, artifact or coding attempt was changed. The separate post-execution
audit recipe and raw/audited-result hashes are retained; the committed unrun freeze
is not rewritten as though results existed before dispatch.

## Disposition

This finite cohort is complete. It again produced the same aggregate 2/10 accepted
and completed-and-accepted counts as the earlier whole-file cohort, but these are
separate public-task observations with different runtime/settings/inputs. Do not
pool them or infer a causal gain or loss. The earlier original trial remains 2/10
artifacts and 1/10 completed-and-accepted; the selected timeout comparison remains
its own 0/2 versus 2/2 result.

No production timeout/default adoption or useful unattended ten-task envelope is
established. Keep feature batches frozen, the original 8/10 target and no-go, and
broader G2, held-out/repeated/two-model G4 and Jorge's G5 decision open. Further work
needs a concrete retained-failure hypothesis or an explicitly proposed narrower
scope with independent acceptance; this closed cohort is not a retry queue.

This docs/data report uses the exact frozen G0 (116 focused/1,453 offline passes,
five live deselected, Ruff/format/mypy and both scripted 90/90 matrices) and the
new retained-outcome audit. Those offline mechanics checks are not model successes.
