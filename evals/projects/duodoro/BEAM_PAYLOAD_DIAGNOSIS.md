# Retained Beam input payload diagnosis

**Offline byte attribution, not a new model run.** The
[diagnostic rows](../../results/2026-10-26-beam-input-payload-diagnosis.json) inspect
all 23 retained Beam request inputs from the [completed cohort](BEAM_COMPARISON_RESULTS.md).
Original inputs, artifacts, criteria, budgets and results remain unchanged. No failed
request was replayed and no model/provider request ran.

## What occupied the retained inputs

All four cells read the repository's AGENTS.md once. That serialized read_file reply
contains 17,387 UTF-8 content bytes, then stays in each subsequent request. This is
one read per cell and repeated history transmission, not repeated AGENTS.md tool calls.
Other file reads and directory listings also remain in history.

At each cell's final actual request:

| Cell | All message payload bytes | Tool-role payload bytes | AGENTS.md reply content bytes | Serialized continuation bytes |
|---|---:|---:|---:|---:|
| T02-B1 | 32,741 | 30,531 | 17,387 | 334 |
| T02-B2 | 40,241 | 31,642 | 17,387 | 5,518 |
| T08-B1 | 36,754 | 33,991 | 17,387 | 466 |
| T08-B2 | 37,807 | 33,991 | 17,387 | 1,519 |

Tool-role payload includes tool-name bytes; reply content excludes that overhead.
Continuation bytes are already included in assistant-role/all-message bytes, not an
additional total. The table excludes tool schemas and per-message token overhead;
the raw diagnostic separately records the existing schema/message token estimates,
saved estimate/output allowance and reported usage for each actual request.
Neither these byte counts nor the heuristic estimates are tokenizer measurements,
backend receipt, caching/billing proof or attribution of model reasoning failures.

In the two refactor cells stopped by total-token gates, tool replies dominate these
message bytes; replay state is much smaller. T08-B1 stops before its next request
at 39,408 reported tokens, while T08-B2 stops after a response at 52,725. The gates
reserve estimated next-request room, so neither needs to reach exactly 64,000.
This record does not loosen those gates or claim the unused nominal allowance was
sufficient to finish. T02-B1's truncation and the two Gemma timeouts retain their
original classification; this diagnosis does not establish their backend cause.

## Bounded follow-up

This supports testing a smaller **opt-in default read window**: preserve whole-file
SHA-256/line counting and exact selected text, return explicit range/continuation
metadata, and let explicit line ranges request more evidence within existing hard
read limits. Existing unrestricted defaults must stay unchanged. Shared runtime,
profile identity and guarded continuation must record a selected window; no client
should invent agent behavior.

Root instructions may matter. A smaller first window is not permission to ignore
remaining instructions, silently discard old history, rewrite replay state, remove
context budget enforcement or alter the original evaluation fixtures. It changes
model-visible evidence and may make completion worse. Deterministic tool/runtime/
restart coverage comes first; any claim of improved task completion needs a separate
reviewed same-model comparison. None is run or authorized by this diagnostic.

## Reproducibility and limits

The recipe validates normalized input hashes against the retained input analysis,
uses the unchanged message_size_bytes/estimate functions, and hashes its recipe and
source sanitized result. The local recipe/receipt are retained at
`/Users/jorge/.local/share/agent-harness/trials/duodoro-beam-payload-8uetyrkx`.
Only counts, allowlisted tool names, input hashes and saved budget metadata are
published; no file contents, tool arguments, raw wire inputs or credentials.
These are retained public-development-task observations, not held-out evidence,
model-quality gains, a supported task envelope or release readiness. Original
cohorts/no-go/defaults/8-of-10/G2–G5 stay unchanged.
