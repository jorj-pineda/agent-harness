# Timer exact-edit failure diagnosis

The selected T02 full-preset failures stopped first on a malformed hash, then
on the preventive total-token gate. They were not two dispatched text mismatches.
In each repeat, the first replace_text supplied 63 lowercase hexadecimal characters
instead of the required 64. Pydantic rejected it before file access or text matching.
The second saved edit record says “Tool not executed: this turn has already stopped.”
It therefore provides no second validation or text-match observation.

The [numeric receipt](../../results/2026-10-24-duodoro-timer-hash-diagnosis.json) records an offline replay of
the first exact arguments against unchanged disposable target copies using frozen
runtime `9f976555fb5b437146b2b07bcc6550e742ab2ff7`. Both registry error strings
matched the saved records exactly; bytes and modes were preserved. The correct
whole-file hash has 64 characters. No hash was corrected and no edit was applied.
Raw arguments/source remain local.

Both saved final stops are post_response_stop: 59,086 prompt plus 757 output tokens
leave 4,157 of the original 64,000. The recorded heuristic prompt estimate is
15,561 tokens, already above that remainder before reserving output. This explains
the recorded prevention without alleging backend tokenizer accuracy or proving
that better validation feedback would have allowed completion. The malformed hash
and escaped text are separate: text matching was never reached in this replay.

All 228 original outcome hashes were checked before and after the probes. The
original cohorts, eight-cell results, unknown usage rows, 8/10 target, no-go and
G2–G5 remain unchanged. No inference, new attempt, repaired artifact, budget change,
Duodoro working-checkout edit or model-quality claim follows.

Retained reporting recipe: `/Users/jorge/.local/share/agent-harness/trials/duodoro-timer-hash-ysait8gd/diagnose.py`.
SHA-256: `b34eec991d54cae16858c9a79524f4fa2e09f42b7280d86cb9bce226c138931b`.
The recipe imports the frozen runtime and invokes only the real tool registry;
it constructs no provider. Its fixed repository report output must be redirected
to a fresh path for reproduction rather than overwriting published evidence.

Next bounded work: make malformed expected_sha256 feedback explicitly request the
complete latest read_file hash, while preserving the schema, strict rejection,
stale-byte/mode guards and budgets. Scripted correction/restart and budget-stop
regressions can validate mechanics; a live benefit would need a separate reviewed
experiment. Do not pad, truncate, lowercase or infer the model's hash automatically.

## Feedback follow-up

Malformed string hashes now receive the complete-hash copying instruction before
workspace resolution. The exposed JSON schema is byte-structure identical to the
frozen input schema; valid hashes still face all current-file/content/mode guards.
There is no automatic correction or new retry allowance. Both retained first
arguments reach the new feedback even in an empty temporary workspace. This is
an offline validation of diagnostics, not a successful live correction or edit.
64 focused and 1,461 offline tests (five live deselected), static checks and both
scripted 90/90 matrices passed. Scripted API recovery/budget/restart checks are the
next separate slice; no new live experiment is authorized by this diagnostic.

## Scripted API recovery and prevention coverage

The HTTP/SSE regression uses scripted responses, real file/check tools and fresh
SQLite lifecycle resources. It reads the current hash, submits one malformed hash,
retains the diagnostic in model history, then proposes an explicit corrected hash.
With budget available, exactly one replacement preserves existing user bytes/mode,
the configured pytest check passes and the scripted turn completes. In the separate
synthetic over-reported-usage branch, the existing post-response total-token gate
prevents the corrected edit before dispatch; the artifact is unchanged and checks
remain not_run. This exercises an overrun stop, not the exact retained T02
follow-up estimate boundary or a strict backend token cap.

Both branches preserve their original errors, final response and budget journal
across a fresh API lifecycle. Saved GET and explicit same-key replay return identical
records without inference or tools, even after later user content is added. The
unrelated protected file, permissions and visible check bytes remain unchanged.
The scripted provider chooses the correction; this does not demonstrate Gemma
recovery or independent task acceptance, and creates no new retry allowance.

Validation: four new transport/budget cases, 69 focused tests and 1,465 offline
tests passed (five live deselected), plus Ruff/format/mypy. The existing Chroma
warning remains. This test/docs-only slice does not rerun the scripted matrices;
both 90/90 results in the preceding feedback slice retain their original scope.
No inference, historical result/criterion change, new live trial, Duodoro edit,
quality gain or G2–G5 closure. This completes the three-PR diagnosis/feedback/
scripted-regression batch; the original no-go and feature freeze remain.
