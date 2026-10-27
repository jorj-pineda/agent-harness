# Offline replay of the source token gate

**2026-10-27 — No new inference or runtime change.** The
[source-wait cohort](SOURCE_WAIT_RESULTS.md) stopped after inspection in every cell.
The [retained replay and payload attribution](../../results/2026-10-27-token-context-diagnosis.json)
reproduce the exact shared budget decision for all four saved runs. All 3,627
original outcome hashes match before and after analysis; budget/loop code matches
the observed runtime `4d57cc05fdfea5b33027af9b9aa9a074195320e9` byte for byte.

## The actual budget boundary

Every source consumed 26,325 reported prompt and 156 output tokens, leaving 5,519
of its 32,000-token allowance. Its last request alone reported 9,845 prompt tokens.
The shared post-response gate needs at least that previous prompt lower bound plus
256 output tokens: **10,101 > 5,519**. It therefore stops before executing the
proposed `emit_plan`. The saved reason and all four request allowances/estimates
match deterministic replay through `TokenBudget`.

This decision uses the last reported prompt count, not the byte heuristic. Changing
the heuristic ratio alone cannot alter this particular decision with the same
reported prefix. The context-window cap and the cumulative turn-token cap are
separate: free context-window space does not replenish the remaining turn budget.
The check is a lower bound for retaining the same history, not a measurement of
an invented fifth request. No fifth request or model response was sent.

## Repeated message payload

| Request | Reported prompt tokens | Message payload bytes | Tool-reply payload bytes |
|---|---:|---:|---:|
| 1 | 1,963 | 1,699 | 0 |
| 2 | 6,303 | 19,125 | 17,396 |
| 3 | 8,214 | 25,338 | 23,564 |
| 4 | 9,845 | 30,418 | 28,594 |

The shared `message_size_bytes` metric counts content, tool names/arguments and
continuation fields. It omits wire framing and counts schemas separately. Across
the four sent requests, tool replies account for 69,554 of 76,580 message payload
bytes. These are **payload bytes, not provider tokens, billing or total wire size**.
Each source has the same counts; repeats are not independent samples.

The AGENTS.md reply's serialized content is 17,387 UTF-8 bytes and remains in
requests 2–4. The parser reply is 6,159 bytes, retained in requests 3–4; the test
reply is 5,021 bytes in request 4. Each tool-reply payload additionally counts the
nine-byte `read_file` name. Full file hashes and observed line ranges remain in
the evidence. Their presence establishes retained read data, not comprehension,
correct instruction precedence or accepted code. No source body is published here.

## Consequence for the next experiment

This is evidence of repeated history cost and a reproduced preventive stop, not
an estimator defect or a demonstrated compaction benefit. Omitting historical
assistant/tool messages can also discard repository rules and prior agreements.
A projected fork retaining user messages and selected saved spans must still use
the shipped review/admission guards and explicit omission/new-budget acknowledgments.
Smaller inputs alone cannot establish task acceptance or preserved meaning.

A justified next measurement is fresh native full-history continuation versus
explicitly reviewed projected recovery from new naturally bounded sources, with
source cost charged to both policies and unchanged per-workflow limits. It needs
its own protocol, original independent criteria, actual input guards, reviewed
freeze and parent approval; these completed source-wait cells remain closed.
No history edit, unchecked workspace mutation, budget increase, automatic resume,
default change or new inference follows from this analysis.

Five negative probes alter an input, omit an input, remove reported usage, alter
the stop reason or mark budget evidence truncated; all refuse. This is a bounded
replay of this retained four-request shape, not a generic parser or new runtime
policy. Private script/output are retained under the JSON's analysis path/hash.
Original cohorts, no-go, 8-of-10 target and broad G2–G5 remain unchanged.
