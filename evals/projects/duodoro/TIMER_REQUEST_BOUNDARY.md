# Returned versus timed-out timer request boundary

**Offline comparison complete; no inference.** The four cells in the
[hash-feedback comparison](HASH_FEEDBACK_RESULTS.md) timed out on request two,
where two earlier full-preset T02 attempts returned a response. Neither earlier
T02 artifact was independently accepted: “returned” refers to one provider call,
not a successful coding task.

The [numeric receipt](../../results/2026-10-25-duodoro-timer-request-boundary.json)
compares twelve retained normalized/wire pairs from six runs. It preserves all
332 original outcome hashes and the bytes of the six selected session databases.
No provider, network request, input replay, patch repair or new acceptance ran.

## Observed boundary

| Cohort/cells | Request-one wire bytes / estimate | Request-two wire bytes / estimate | Request-two observation |
|---|---:|---:|---|
| Prior T02-F1/F2 | 9,462 / 3,116 | 27,325 / 8,933 | returned; reported 6,304 prompt/25 output each |
| Later C1/T1/T2/C2 | 9,464 / 3,117 | 27,327 / 8,934 | read timeout; usage unknown |

Every second input follows exactly one read_file(AGENTS.md). The returned content
is the same 16,934 UTF-8 bytes with the same whole-file hash. Both requests have
the same roles/message counts and tool-schema counts. Model identifier, buffered
transport, temperature0, thinking=false, num_ctx16,384 and num_predict2,048 match.
Saved request identities/allowances and original owner-only file permissions match.
These byte/heuristic sizes do not attest backend tokenization, memory use or cause.

The diagnostic compares prior F1 with later C1/T1 and prior F2 with later T2/C2,
for requests one and two: eight pairwise comparisons. **Actual raw wire bytes
differ.** In a detached in-memory copy only, replace the single exact trailing
system line `Workspace root: <that cohort's literal root>` with one common marker.
After that one explicit substitution, every parsed JSON wire body is equal.
Nothing else is removed, normalized, summarized or changed. No substituted request
is forwarded, saved over an original or called experimentally equivalent.

Four negative probes change user text, tool description, output allowance or other
system text. The diagnostic still detects all four differences; its workspace-line
substitution cannot mask them. Original normalized records keep their generated
call IDs and hashes; this comparison does not rewrite history to align them.

This supports a narrow observation: no unexpected schema/option/user/read-content
drift is shown at this boundary. The extra two bytes and one heuristic token are
accounted for by the explicitly different root line. It does **not** show that the
path caused the failure, that the backend received or tokenized everything as
expected, or that host/model/server state was equivalent across dates/runtimes.
Those states were not instrumented sufficiently for attribution. Reducing context,
increasing timeout or changing a model is not a proven remedy from this comparison.

## Preservation and disposition

Original artifacts, input files, databases, criteria, runtimes/defaults, budgets
and every prior result remain unchanged. Raw source, requests and responses stay
local. The recipe and receipt are retained at:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-timer-request-boundary-hmrj56lx`

The structured receipt pins the recipe hash and both source report hashes. Its
write-once repository output refuses replacement; reproduction must redirect to a
fresh report path. The recipe only reads source evidence and writes new diagnostics.
Its four negative probes are synthetic offline comparison checks, not model tests.

The finite failed-feedback cohort stays closed. Original ten-task counts, the 8/10
target, no-go/feature freeze, production defaults and G2–G5 stay unchanged. This
observation is not another coding attempt, quality gain or release progress.
A separately scoped intended-model comparison still needs current authorized
access, comparable fixed inputs/limits, private recording support and independent
acceptance; it must not reuse these failed cells as retries.
