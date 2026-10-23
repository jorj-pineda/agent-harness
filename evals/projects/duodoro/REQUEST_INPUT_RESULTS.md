# One-attempt request-input diagnostic

After independent review and PR #190's merge, the one frozen original T09 CLI
attempt ran at `1a0573e7faa42e045cdd18e108e22d035ecf0e96`. **0/1 accepted artifacts;
0/1 completed-and-accepted.** The fifth call failed with `read_timeout` after
60,046.100 ms of observed adapter HTTP-post time. No file changed. The attempt
was not replaced or repaired, and there was no live replay of its recorded input.

The [freeze](REQUEST_INPUT_FREEZE.md) remains unchanged. The
[structured results](../../results/2026-10-23-duodoro-request-input-diagnostic.json)
retain the outcome, IDs, independent grades/regrades, budget/progress evidence,
input hashes and safe structural analysis. Raw normalized histories/wire bodies
remain local plaintext in private files, not published here.

## Exact inputs now retained

Five normalized/wire pairs match all five saved budget request identities,
allowances and message counts, with no prefix truncation. Every wire body matches
production adapter serialization byte for byte when reconstructed **offline with
MockTransport**. This is not a second live request or a backend replay. Capture
before delegation proves prepared input, not backend receipt or correct semantics.

| Request | Messages | Wire JSON bytes | Prompt-token estimate | Output allowance |
|---|---|---|---|---|
| 1 | 2 | 8,666 | 2,843 | 2,048 |
| 2 | 4 | 8,871 | 2,898 | 2,048 |
| 3 | 6 | 17,539 | 5,306 | 2,048 |
| 4 | 8 | 22,165 | 6,718 | 2,048 |
| 5, timed out | 10 | 22,905 | 6,919 | 2,048 |

The configured model, buffered stream flag, thinking-off flag, temperature,
16,384 context and 2,048 request output remain unchanged across all five wire
bodies. Tool specifications are identical. All retained tool-reply IDs/names
correspond to their preceding assistant calls; roles and exact normalized prefixes
remain continuous. These checks find no observed schema/option drift or continuity
break at this boundary, not proof that every backend semantic requirement is met.

The first response was empty with zero tools, followed by the existing completion
recovery. Subsequent completed calls ran tree, read_file and emit_plan. Complete
untruncated progress records their ends without errors; no write or command check
was dispatched. Failed session archives do not become a restored editing state.

## What changed after the plan

Request five extends request four's exact normalized prefix by two messages: the
assistant's one emit_plan call and its tool result. The wire body grows **740
bytes**, the normalized record **851 bytes**, and the prompt estimate **201
tokens**. The plan has two steps (74 and 105 UTF-8 bytes); no step text is published.
Its result matches the pinned implementation's steps/summary/step_count fields,
step cleaning/count and summary. The repeated plan values are expected tool
output, not evidence of an unexplained expansion or serialization corruption.

The failed request's 6,919-token estimate plus 2,048 output allowance is **8,967**,
below the configured 16,384 context limit. This is the harness's byte heuristic,
not the backend tokenizer, actual context occupancy or proof that the backend
honored those limits. It does not rule out context-related backend behavior.
The observed small plan delta and unchanged options do not support the proposed
unexpected-expansion/option-drift explanation in this attempt. A broader claim
that the request cannot be malformed is not established by these checks.

## Failure and measurement limits

Four returned responses report known usage: the prefix is **13,540 prompt / 138
output tokens**. Aggregate usage remains **unknown**, including the fifth in-flight
request. Prefix totals are not zero usage, billing evidence or a known within-
budget success. The failed saved run has no final response, so completion/check
fields remain unrecorded. No final answer or check status is invented from progress.

Normalized capture I/O is outside adapter HTTP post; wire capture I/O is inside
it. The observed 60,046.100 ms includes wire recording and transport waiting, not
pure inference. Synchronous fsync can delay cancellation; the 300-second dispatch
allowance is not a hard return deadline. Source API lifecycle lasted about 132.2
seconds including startup/shutdown, separately from the failed-call measurement.

This recorder does not collect failed responses, first-byte/body-chunk timing or
backend logs. Those remain unknown for this attempt. The earlier
[transport cohort](TRANSPORT_COMPARISON_RESULTS.md)'s zero-chunk observations stay
separate; do not copy them into this new run. Earlier normalized replies naturally
appear in later input histories, but no raw provider responses or failed output
were captured. No inference/generation/queue cause follows from input validity.

## Preservation, restart and cleanup

Exactly one durable run and chat submission used the retained printed key. After
settlement, the source API shut down fully and Gemma was unloaded. The unchanged
full artifact was archived and independently graded twice, reproducing rejection;
protected bytes/modes, settings, criteria/reference and full manifest/diff hashes
match. Guarded review restart performed only CLI GET inspection and returned
identical saved JSON. It allocated no new input recording and made no model call.
SQLite saved records and tool progress match the retained review evidence.

All ten private input file hashes and owner-only permissions match: 163,899 total
stored bytes, below the recording limits, with no record clipping or bound stop.
All frozen recipe/source/preflight hashes and 24 retained outcome/report evidence
hashes match. The raw local results and unrun committed preparation manifest are
unchanged. Original cohorts and Duodoro head/status/index/refs are preserved.

Both owned APIs exited with complete shutdown logs, port 18759 is released and
the loaded-model list is empty. The existing Ollama service remains. No paid/new
model, dependency installation, Docker action or Duodoro working-checkout edit.

No execution correction or coding rescue occurred. One post-execution helper
incorrectly expected emit_plan result keys status/steps. Reading the pinned tool
implementation corrected that assumption to steps/summary/step_count and its
normalization rules. The original helper is retained; the final read-only audit
passed without changing any raw input, artifact, execution recipe or model attempt.
This is a reporting correction, not an input anomaly or another coding attempt.

Local immutable artifacts, private inputs and recipes remain under:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-request-input-v14rnqm5`

## Validation and disposition

The frozen launcher completed its one submission, grade/regrade and restart gates.
The frozen offline analysis matched every input pair and production mock wire;
a separate read-only audit checked plan-boundary structure, recorded identities,
SQLite, all hashes/permissions, unchanged artifacts and process/model cleanup.
This docs/data report does not rerun preparation's 116 focused/1,453 offline/static
checks or two scripted 90/90 matrices as task-quality evidence.

The input-observation gap is closed for this one failure. The timeout is not fixed,
and its backend cause remains unknown. No unexpected post-plan conversion,
expansion or configured-option change was observed in the inspected records.
Further work needs a separately bounded hypothesis or explicit narrower envelope;
these captured inputs are not authorization for extra model replay or automatic
retries. No production/default/limit change or coding-quality improvement follows.
Keep the original cohorts, no-go and G2–G5 unchanged.
