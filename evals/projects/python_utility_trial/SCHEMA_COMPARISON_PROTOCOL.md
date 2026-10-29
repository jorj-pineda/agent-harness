# Replacement schema comparison protocol

**Prepared; all eight rows unrun.** This is the first of three requested PRs:
protocol, independently reviewed execution freeze, then conditional results.
The [fixed plan](schema-comparison-plan.json) describes a new local comparison;
protocol merge alone cannot authorize generation. No old attempt is resumed.
Original Duodoro counts, the 8/10 target, release no-go and G2–G5 remain intact.

## Hypothesis and fixed scope

The [offline replay](TOOL_REFUSAL_REPLAY.md) reproduces nine newline-edit refusals
despite specific corrective messages. The candidate in #267 adds preventative
multiline examples and once-encoded JSON guidance to `replace_text`'s old/new
parameter metadata. Measure whether this schema bundle changes accepted coding
completion, rather than assuming it prevents errors. This does not isolate the
effect of examples from descriptions or their additional input size.

Use only original public `divide_zero` and `stable_dedupe` requests, fixtures,
visible tests, external criteria and references at source revision
`6be17d2ee142c0a36c4b69e73a95e558e8af43e1`. Both showed repeated encoding errors
in the closed trial. This failure-selected sample is not held out or a broad
coding envelope. No parser, command-argument intervention or other task is added.

Control C uses runtime `a563a104b82993202792898ebde2a06c3ecb822a` (pre-example);
E uses `530abdca8aa65e17f42139ca31b9df72ad3022f8` (example candidate).
Revision paths differ only in `tools/code.py`, its tests and the roadmap.
The later freeze must independently prove that executable runtime behavior and
base prompts are equal except the two replacement fields' descriptions/examples,
and that validation, edit functions, tool inventory and other schemas are equal.
Use actual Git archives of both revisions, not a hand-reconstructed schema or
an unchecked production override. Freeze helper/environment compatibility for both.

| Order | Row | Task | Repeat | Arm |
|---:|---|---|---:|---|
| 1 | SC01 | divide_zero | 1 | C |
| 2 | SC02 | divide_zero | 1 | E |
| 3 | SC03 | stable_dedupe | 1 | E |
| 4 | SC04 | stable_dedupe | 1 | C |
| 5 | SC05 | divide_zero | 2 | E |
| 6 | SC06 | divide_zero | 2 | C |
| 7 | SC07 | stable_dedupe | 2 | C |
| 8 | SC08 | stable_dedupe | 2 | E |

Each row has a fresh owned root, independent initially absent stores and a fresh
owned backend lifecycle. Send the original request once through actual CLI/local
HTTP and the shared application runtime/tools. There is one phase per row, no
history/source sharing, continuation, manual patch repair, retry, replacement or
budget transfer. Success does not skip another scheduled repeat. Retain failed,
blocked and unrun rows; never rank unequal partial cohorts as a complete result.

## Same model and allowances

Only local Ollama `gemma4:12b`, digest
`4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`,
expected server 0.33.2. Fresh endpoint/version/model/image verification is required;
historical pins do not attest a current service. No hosted/paid calls, downloads,
new models, dependency installs, Docker or warmup generation. Never stop/unload
the user's 11434 service or change its identity/configuration. Only separately
owned API/backend processes may be managed; inspect current listeners first.

Each row reserves 150 dispatch seconds, 32,000 total tokens, 4,096 output tokens,
12 tools and six iterations. Per-request output is at most 2,048, minimum 256;
all four provider waits are 180 seconds, CLI response inactivity 600. Temperature
zero, thinking off, explicit context/Ollama context 16,384, ratio 3, compact/full,
read window 0, one completion correction, identical-call limit 2, file budget 2,
verification required and no required plan. Root AGENTS injection/support are off.
The exact configured check is `python -m pytest -q test_visible.py`.

Each arm reserves 600 dispatch seconds, 128K total, 16,384 output, 48 tools and
24 iterations. Physically eight attempts reserve 1,200 dispatch seconds, 256K
total, 32,768 output, 96 tools and 48 iterations. These are sums, not pools or
hard end-to-end deadlines. Setup, hashing, snapshot, final storage, grade/regrade,
restart and cleanup overhead remain separate. Keep dispatch attempts, raw tool
records, pre-dispatch refusals and conservative execution counts distinct.

## Independent acceptance

Original external tests AND independent source review must pass for artifact
acceptance. Native completed status is additionally required for the primary
completed-and-accepted metric. Editable checks or final prose cannot redefine
acceptance. Use the prior frozen rules without expansion after seeing patches:

- D1: general real division including nonintegral results and Python zero-division
  errors. Floor division and example-specific results fail.
- U1: repeated integer removal preserves first-appearance order; sorting or
  unordered set output is insufficient.
- C1: reject example-specific hardcoding, weakened/deleted visible tests,
  unrelated edits and direct violations of the original request. Do not require
  reference text or implementation style.

The original division test gap and floor-division control remain relevant; an
external pass alone is provisional. Preserve pass/reject/unknown reasons per rule
and artifact hashes. Unknown review is not pass. Review occurs outside the coding
loop, may finish after the cohort, and cannot guide another model call or edit.
Independent exact-content review must check the verdicts before results merge.
No premium supervisor is introduced into the product or awaited during dispatch.

## Freeze, preservation and refusal requirements

Before generation, the second PR must bind both full runtime trees/archives,
all executable recipes and preparation evidence, task contracts, model image
and originals, actual interpreter/git/node/uv locations, root identities and
environment. Use independently hashed APFS CoW model copies, not hardlinks.
Never evaluate in a working repository or commit/push evaluation patches.
Preserve Duodoro head/status/index/refs and user configuration/identity metadata
without reading credential or identity contents.

Capture each row's first normalized and native wire inputs through the production
adapter with MockTransport, without forwarding inference. Cross-arm/task-pair
comparison may differ only by the declared workspace-root line and exact C/E
replacement schema metadata; every other message/schema/setting/allowance must
match. Same-arm repeats may differ only by declared workspace-root line. Actual
forwarded hashes include full paths and the actual schema; no normalization is
used to force equality. The intervention's schema size can affect later heuristic
allowances; preserve that as part of the measured bundle rather than equalizing it.

The complete operator must check first inputs before forwarding and guard every
wire route/model/temperature/context/thinking/timeout/actual output allowance.
Validate original raw returned usage as exact nonnegative integers before loop
delivery, preserving malformed values/presence; booleans/string/float counts fail.
An output report above its paired request's actual allowance stops the cohort,
even if below the configured maximum or phase totals.

Require exact-runtime focused/full/static G0 and both labeled scripted matrices
for both archives; full eight-row actual CLI/API scripted rehearsal; per-arm
input/schema drift refusals; reduced-allowance and malformed-usage probes;
failed-check, wall-stop, uncertain POST and lookup-failure preservation; ownership
registration/receipt failures and closed-store GET-only restart. Simulation fields
must be explicit and forbidden in live configuration. Preparation proves mechanics
only; metadata backend checks cannot prove generation cleanup or coding quality.

Each row must settle tools/resources, preserve its eligible bytes/modes and
original grade/regrade, archive stores and checkpoints, and verify prior artifacts
and protected criteria before another allocation. Unknown changes, special entries,
drift or incomplete preservation refuse. Closed store clones must preserve native
run/archive identities and actual-store byte hashes. API/backend children must
exit zero with registered groups absent/listeners free; managed tools must be
terminal while retaining real failed/cancelled outcomes, not necessarily exit zero.
These checks do not create OS isolation or discover arbitrary detached workers.

Require independent exact-head Sol review of the entire final freeze/private
bindings, both named CI checks, merge, exact parent-observed review and approval
receipts, and fresh base/model/resource/workspace gates before generation and
between rows. No old approval or checkpoint grants permission. Existing sources
or a stop marker cannot be resumed. An uncertain submission permits original-key
GET only, never another POST, new key or generic-session fallback.

Finalized native policy/token/context/tool/iteration refusals and exact configured
150-second native wall-budget outcomes may proceed only after preservation,
grading and clean settlement. `CancelledError` alone is insufficient. Unknown
in-flight wall usage stays unknown. Any provider/transport failure, malformed or
missing returned usage, reported request/phase overrun, unexplained cancellation,
runtime/environment/model/input/criteria drift, inconsistent grade, preservation
or cleanup failure stops all remaining rows. No operator repair or substitution.

## Reporting and present preparation

Primary counts are completed-and-independently-accepted /4 per arm and /2 per
task/arm. Keep artifact acceptance, configured checks, source review, native stops,
known budget compliance and interventions separate. Supplement with observed
encoding errors, successful edits, requests, tool/iteration records, payload bytes,
estimated versus reported tokens, elapsed scopes and unknown usage prefixes.
The smaller schema is not presumed cheaper overall, and an error reduction is
not accepted completion. Preserve all raw counts and failed/blocked/unrun rows.

Fresh offline assertions verify both selected originals fail visible/external
checks and references pass; all 42 original catalog bytes/modes match. The plan
binds that receipt and its driver. No input capture, final operator, model gate,
G0 for both archives, execution approval or inference is claimed by this protocol.

Two failure-selected public tasks and temperature-zero repeats cannot establish
general quality, statistical independence, held-out gains or a supported envelope.
Fresh backend processes do not control hardware/cache/timing. This small matched
comparison can report observed differences under the pinned schema bundle, not
prove universal causation. Defaults, original cohorts, no-go/8-of-10 and broad
G2–G5 remain unchanged. The conditional results PR authorizes no further attempts.
