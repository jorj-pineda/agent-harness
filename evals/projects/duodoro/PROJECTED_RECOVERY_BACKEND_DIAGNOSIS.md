# Offline diagnosis of the stopped source's backend progress

**2026-10-27 — No new inference or configuration change.** The
[stopped comparison](PROJECTED_RECOVERY_RESULTS.md) retains a failed second call,
unknown aggregate usage and no C/P result. Its evaluation-owned Ollama log supplies
additional backend progress evidence. The
[allowlisted chronology and evidence hashes](../../results/2026-10-27-projected-recovery-backend-chronology.json)
were extracted offline; all 894 public cohort evidence hashes still match.

The failed call was not idle for its entire wait. The second backend task logs
prompt processing reaching progress 1.00, then generation counters 100, 140 and 180,
before the HTTP 500/cancel sequence. This narrows the failure description from an
unlocalized read timeout to observed prompt processing and generation without a
completed buffered response. It does not identify why a valid terminal response
was not available by the client's timeout.

## Observed chronology

The owned log contains exactly two chat POST records, two runner task launches and
one loaded model. Task 0 precedes the first successful POST; task 27 precedes the
second failed POST. This is an ordinal association within the sole owned cohort,
not cross-layer request-ID linkage. The first task's 1,964-token prompt and 22-token
terminal generation count agree with the returned provider usage; its roughly
24.552-second backend POST duration also agrees with the adapter's 24.571 seconds.
No additional generation or warmup was sent through this endpoint.

For task 27, the runner reports a 6,304-token prompt with a 16,384-token context.
It selects a cached prefix and logs prompt-processing progress through 4,340
processed tokens at progress 1.00 and backend-relative elapsed 37.69 seconds.
The first call's separate terminal prompt-evaluation summary is 16,389.40 ms;
there is no corresponding final timing/usage summary for task 27. These fields
have distinct scopes and must not be substituted for each other.

After prompt processing, task 27 logs `n_gen` 100, 140 and 180 at approximately
13.2 tokens/second. The HTTP log then records a 500 response at one minute and
cancellation of task 27. The adapter independently records a read timeout after
60.007 seconds with no returned response. Untimestamped runner progress is not an
end-to-end clock: subtracting 37.69 from 60.007 would be arithmetic, not a measured
remaining generation allowance. No generated text or eventual completion is retained.

The source's first tool read AGENTS.md lines 1–265 of 265; the second captured
request carries that history. The larger second input and additional prompt work
are observations, not a controlled causal comparison. Cache counters and runner
prompt/generation counters do not replace terminal provider-reported usage. In
particular, `n_gen = 180` does not turn the failed call into a known 180-output-token
response or repair the unknown aggregate total.

## What follows, and what does not

The evidence excludes a description in which this owned second task made no prompt
or generation progress before cancellation. It remains compatible with useful,
unhelpful, malformed or incomplete generated content; the buffered adapter received
none of it as a completed response. A timeout increase, smaller instruction reply,
streamed transport or another context policy has no demonstrated benefit here.
Earlier timeout/streaming/read-window comparisons retain their own outcomes and
cannot be silently generalized into an approved remedy for this stopped cohort.

The parser requires the two POST/task sequence, completed prompt-progress marker,
three generation markers, expected ordering, cancellation and absent second terminal
summary. Five negative probes remove a POST, cancellation, complete prompt marker,
generation markers, or add an extra POST; each refuses. This validates this bounded
extraction, not a general Ollama log parser. The analysis script and raw log hashes
are published; exact allowlisted lines retain original line numbers. Source, model
files, recipes, criteria, original outcome rows and evidence were not modified.

This completes one offline failure investigation, not a new launch protocol or
feature. No further inference is authorized by it. A next experiment needs an
explicit hypothesis and separately frozen scope; these unrun rows remain closed.
No correction to the original result, known-usage claim, quality gain, default
change, useful unattended envelope or no-go/8-of-10/G2–G5 closure follows.
