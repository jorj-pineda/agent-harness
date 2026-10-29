# Replacement schema comparison results

**Complete: control 0/4 accepted artifacts and completed-and-accepted;
example arm 0/4 accepted artifacts and completed-and-accepted.** All eight
implementations remain unchanged and incorrect. Every run reaches its six-iteration
limit; no native completion occurs. This fixed comparison demonstrates no observed
accepted-completion benefit from the description/example bundle. It does not
establish general equivalence, absence of benefit elsewhere or release readiness.

The [protocol](SCHEMA_COMPARISON_PROTOCOL.md), [freeze](SCHEMA_COMPARISON_FREEZE.md)
and [sanitized evidence](schema-comparison-results.json) retain the eight fresh
rows. Independent three-round Sol review, both CI checks, freeze merge
`13579d399fd340df5005345d0473684ba23a388f`, parent receipts and fresh launch gates
preceded generation. No retry, replacement, manual patch repair, extra attempt
or actual-cohort cleanup intervention occurred. The operator exits zero with all
eight rows settled and no cohort-stop marker.

## Fixed conditions and outcomes

Control C uses archive `a563a104b82993202792898ebde2a06c3ecb822a`; E uses
`530abdca8aa65e17f42139ca31b9df72ad3022f8`. The only executable/schema difference
is the replacement old/new fields' descriptions and multiline examples. Original
requests, source fixtures, criteria and references remain unchanged. Both arms
use local Ollama 0.33.2, `gemma4:12b` digest
`4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`,
thinking off, temperature zero and the same 150-second/32K/4,096-output/
12-tool/six-iteration allowances. No budget transfers or pooled reservations.

| Row | Task | Arm | Native stop | Artifact |
|---|---|---|---|---|
| SC01 | divide_zero | C | Iteration limit | Reject |
| SC02 | divide_zero | E | Iteration limit | Reject |
| SC03 | stable_dedupe | E | Iteration limit | Reject |
| SC04 | stable_dedupe | C | Iteration limit | Reject |
| SC05 | divide_zero | E | Iteration limit | Reject |
| SC06 | divide_zero | C | Iteration limit | Reject |
| SC07 | stable_dedupe | C | Iteration limit | Reject |
| SC08 | stable_dedupe | E | Iteration limit | Reject |

Each task/arm is 0/2 accepted and 0/2 completed-and-accepted. All planned rows
remain in the denominator. There are no failed-infrastructure, blocked or unrun
rows; settled bounded failure is not a successful coding workflow.

## Acceptance and observed tool behavior

Original external double grades and independent reconstruction regrades reject
all eight artifacts. Post-cohort exact-source review also rejects D1/U1 and C1:
division still returns constant `0.0`, violating general real division and zero
division errors; deduplication still returns `list(set(items))`, violating
first-appearance order. Visible tests, eligible files and modes are unchanged.
The review uses original-request behavior and concrete counterexamples, not
reference-text similarity or filename overlap. Independent exact-content review
must confirm these verdicts before this results PR merges. No review guided a
later request, edit or retry.

Control records eight `replace_text` attempts, all refused for literal backslash-n
characters where the file has line breaks. Treatment records six such attempts
and six identical refusals. SC08 attempts no replacement at all, so the smaller
error count is not evidence of repaired encoding or a successful edit.

SC08 first supplies a quoted JSON command array, which validation rejects. It then
resends the proper configured tokens, runs the visible check and gets exit 1 on
the unchanged unordered output, before emitting a plan at the iteration boundary.
Its saved verification status is `failed`; the other seven are `not_run`.
This command correction is one observed within-phase interaction, not a tested
command-feedback intervention or artifact rescue. No eligible implementation or
visible-test bytes/modes changed; ignored check caches are outside that claim.

## Usage and timing scopes

All 48 request attempts have paired normalized/wire inputs and returned response
observations. All raw usage counts match their pre-adapter records. No unknown
usage tail or provider failure was observed; counts remain provider reports,
not billing or independent backend-tokenization attestation.

| Metric | C | E |
|---|---:|---:|
| Reported prompt tokens | 50,654 | 51,438 |
| Reported output tokens | 1,732 | 1,395 |
| Requests | 24 | 24 |
| Tool records / conservative dispatch upper bound | 24 / 24 | 24 / 24 |
| Replacement attempts / newline refusals | 8 / 8 | 6 / 6 |
| Executed configured checks | 0 | 1 (failed) |

All per-row known reported token/output/request counts and conservative dispatch
bounds fit their configured limits; saved creation-to-finalization times are below
150 seconds. This does not turn rejected artifacts into budget-successful coding.
Tool record counts can include earlier preconditions; the dispatch number is an
upper bound, not a successful-tool count. No native “not executed” stop records
occur in this cohort.

The summed saved run times are 241.7 seconds for C and 220.7 for E. They include
setup/storage overhead and are not dispatch-only or a causal speed comparison.
The total operator timestamp interval is 1,112.7 seconds, including repeated
hash gates, allocation, grades, restarts, storage and cleanup. The 1,200-second
dispatch reservation is not a hard end-to-end deadline. Hardware, cache and
timing remain uncontrolled despite fresh owned backends.

## Audit and limitations

The post-cohort audit reconstructs eight artifacts, matches final native
checkpoints and reproduces independent rejection. It verifies all 48 input pairs
and row-specific first hashes/profiles, raw counts, one original chat POST per
row, closed standalone stores, saved native run/archive equality and GET-only
restart receipts with actual store bytes unchanged. Eight arm-worker receipts
match the declared runtime and parent ledger.

All 24 actual API/backend lifecycles exit normally with code zero; workers and
parent are reaped, registered groups are absent and owned listeners are free.
Both frozen runtimes, model images, original/prepared evidence, environments,
Duodoro head/status/index/refs and user-service/configuration/identity metadata
match their pins. User service 11434 is untouched. These are finite ownership
checks, not OS isolation or discovery of arbitrary detached workers.

Retained root:
`/Users/jorge/.local/share/agent-harness/trials/schema-comparison-live-p385qic1`.
The public JSON matches both private summary copies and binds 545 outcome files,
including the executed audit, source-review recorder, builder, approvals, actual
run data and reconstructions. The merged freeze separately binds preparation,
both runtime trees and model images. Its failed scripted parent-shutdown probe
and manual owned-backend cleanup remain preparation evidence, not an actual
cohort attempt or normal-settlement receipt. Summary self-references and the
mutable handoff journal are explicitly excluded. Audits ran no inference.

Two failure-selected public tasks, one model and two ordered temperature-zero
repeats are not held-out or independent statistical samples. This measures the
whole schema-description/example/input-size bundle, not examples in isolation.
It supports no general quality improvement, default-adoption claim, unattended
coding envelope or release closure. Original Duodoro counts, the 8/10 target,
release no-go and G2–G5 remain intact. This closes the requested comparison batch
without authorizing further attempts or changing runtime defaults or budgets.
