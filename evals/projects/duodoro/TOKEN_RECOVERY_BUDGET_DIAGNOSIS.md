# Why the observed recovery arms stopped

**2026-10-28 — Offline replay only; no inference or policy change.** The
[retained analysis](../../results/2026-10-28-token-recovery-budget-diagnosis.json)
replays the twelve recorded requests and final native token decisions for the four
actual arms of the [closed partial cohort](TOKEN_RECOVERY_RESULTS.md). All 8,083
original outcome hashes match before and after analysis. The shared context,
provider types and request-identity code match the frozen runtime byte for byte.
The private script is hash-bound in the public analysis; original evidence and
its public manifest stay unchanged.

## Two different preventive boundaries

The native allowance combines the whole-prompt byte heuristic with the last
reported prompt plus estimated appended messages, taking the larger estimate.
Reported usage is accumulated across requests; available context space does not
replenish the total-token allowance. Each arm reserves 32,000 total tokens and
requires at least 256 output tokens for another request.

| Arm | Reported prompt/output sum | Total remaining | Final prompt estimate | Stop |
|---|---:|---:|---:|---|
| C1 | 23,880 / 115 | 8,005 | 12,585 | Post-response lower bound |
| P1 | 20,051 / 67 | 11,882 | 11,854 | Next-request estimate |
| P2 | 20,072 / 67 | 11,861 | 11,854 | Next-request estimate |
| C2 | 23,880 / 115 | 8,005 | 12,585 | Post-response lower bound |

C's last reported prompt is 9,719 tokens. Even that lower bound plus the minimum
output is 9,975, exceeding the 8,005 remaining. The shared post-response gate
therefore stops before executing the proposed `emit_plan`. Its displayed final
estimate includes the newly appended assistant response, but is not the criterion
that caused this stop. Changing the byte ratio alone cannot clear this boundary
with the same reported prefix.

P's post-response lower bounds still fit: 8,799 + 256 for P1 and 8,806 + 256 for
P2. After executing the third response's read, the next whole-prompt heuristic
leaves only 28 and 7 tokens respectively. Both are below the 256 minimum, so the
next-request gate refuses. This is a different boundary from C and from the
previous [source-wait replay](TOKEN_CONTEXT_DIAGNOSIS.md). No fourth request was
sent; its actual token count and behavior are unknown.

## Sensitivity is not a remedy

A second replay uses the already supported ratio 4 instead of 3, holding recorded
messages, schemas and response usage fixed. C still stops at its post-response
lower bound. P's same saved boundary would admit a request: estimated output room
is 1,805 for P1 and 1,777 for P2, below the usual 2,048 output cap.

This counterfactual does **not** establish safe calibration. Ratio 4 underestimates
all six recorded projected-arm prompts and the third prompt in each C arm. Even
ratio 3 is a heuristic, not a provider-token guarantee. Different allowances or
admission can change later live responses; this replay cannot supply the usage,
patch, check or acceptance of a request that never happened. A smaller estimate
is not itself better coding or demonstrated budget compliance. No default/profile,
limit, history, acceptance criterion or stopped-cohort row is changed.

The recorded payload breakdown keeps tool-reply bytes separate from other message
bytes and schema estimates; none is attributed as provider tokens. Repeated read
history remains a plausible resource obstacle, while the four observed arms made
no edits or configured checks. That does not establish that a narrower read,
projection or estimator adjustment would produce a correct patch or preserve all
repository instructions.

## Evidence and next boundary

The script checks each captured normalized request against its saved archive
prefix and native identity, replays recorded allowances, estimates, cumulative
usage and exact final reason through `TokenBudget`, and refuses five damaged
inputs: truncated evidence, a missing event, changed request identity, missing
usage and changed final estimate. Both the original and sensitivity outputs stay
explicitly labeled; five probes pass. No server, model, dependency install or
store mutation is involved. This is a bounded replay, not a general trace parser.

Unknown cancelled source usage remains unknown: known arm sums cannot establish
source-charged workflow totals. The incomplete cohort has no policy ranking or
observed rescue; original no-go, 8-of-10 and G2–G5 remain open. Any new experiment
must predeclare its intervention and safety/acceptance outcomes, retain comparable
budgets and original criteria, use fresh disposable sources and a reviewed freeze,
and keep every failure/unrun row. This analysis authorizes no generation or retry.
