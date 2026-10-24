# Parser edit-preset comparison results

**Full: 2/4 accepted artifacts and 2/4 completed-and-accepted. Whole-file: 0/4
and 0/4.** Both successes are T08 refactors with actual exact-edit activation;
T02 failed in both presets. This supports the selected refactor hypothesis within
these observed cells, not a general preset ranking or a useful unattended envelope.
Retain no-go, the original 8/10 target and open G2–G5.

All eight [protocol](PARSER_EDIT_PROTOCOL.md) cells ran once, after PR #202's
reviewed [execution freeze](PARSER_EDIT_EXECUTION_FREEZE.md) merged. The runtime was
`9f976555fb5b437146b2b07bcc6550e742ab2ff7`, not these reporting commits. Both arms
used actual CLI, one literal disposable workspace path, independent unseeded
stores, original requests/criteria and unchanged per-cell budgets. They are not
replacements for the original panel cohort. [Structured results](../../results/2026-10-24-duodoro-parser-edit-comparison.json)
retain every cell, input/usage/progress observations, grades, source reviews,
archive and cleanup evidence.

## All eight outcomes

| Cell | Completion | Observed check | Independent acceptance | Edit observation / stop |
|---|---|---|---|---|
| T02-W1 | incomplete | not run | rejected | missing reported output usage; no edit |
| T02-F1 | budget exhausted | not run | rejected | two rejected exact edits; total-token gate |
| T02-F2 | budget exhausted | not run | rejected | two rejected exact edits; total-token gate |
| T02-W2 | incomplete | not run | rejected | missing reported output usage; no edit |
| T08-F1 | completed | passed | accepted | two successful exact edits; independent source review passed |
| T08-W1 | completed | passed | rejected | successful whole-file write; unrelated parser regression |
| T08-W2 | completed | passed | rejected | successful whole-file write; unrelated parser regression |
| T08-F2 | completed | passed | accepted | two successful exact edits; independent source review passed |

Per task/preset, T02 is 0/2 in both arms. T08 is 2/2 full versus 0/2 whole-file.
All eight rows remain in their fixed denominators. No patch repair, replacement,
extra coding turn, new hint or budget transfer occurred. Planned independent
source reviews were review actions, not coding rescues.

T08 full's four successful `replace_text` calls applied the helper/refactor and
export without whole-file fallback. Both resulting helpers preserve the original
Number/default/clamp expression, both duration fields reuse the same helper, and
duplicated inline expressions are removed. The original intention whitespace
regex and other parser bodies remain intact. Each has an incidental blank line
in the edited export block, with no unrelated semantic change or test weakening.
Executable acceptance, regrade and required source review pass independently;
the model's answer text was not used to accept the artifact.

Both T08 whole-file artifacts introduce/export/reuse the helper but change
`/\s+/g` to `/\\s+/g` in unrelated `parseSetIntention`, alongside whitespace-only
churn. Frozen external behavior checks and source reviews reject the regression.
Their passing visible helper check and completed state remain observations, not
semantic acceptance or proof of every answer claim. Neither artifact was repaired.

T02 full's four rejected exact-edit records across the two repeats applied no
mutation. The model did not reach a check before the preventive total-token gate.
The targeted-edit mechanism therefore did **not** activate successfully on T02.
Both T02 whole-file final responses omitted usage and proposed no normalized edit;
their usage totals stay unknown. These failures remain separate from T08's success.

## Usage, timing and comparison limits

There were 54 recorded requests and 54 returned responses. Only T02-W1/W2 have
unknown aggregate usage; the known prefix does not become the task total. Failed
request-cap accounting is not assigned zero. Provider reports are not billing or
backend token-compliance attestation.

| Cells | Reported prompt / output tokens per cell | Saved loop latency, seconds |
|---|---:|---|
| T02-F1 / F2 | 59,086 / 757 each | 199.86 / 197.18 |
| T08-F1 / F2 | 45,348 / 1,007 each | 158.93 / 158.70 |
| T08-W1 / W2 | 37,366 / 2,208 each | 252.38 / 252.97 |
| T02-W1 / W2 | unknown | 254.59 / 249.37 |

The two accepted full cells have known reported usage within nominal token and
request/iteration allowances. This does not attest backend enforcement or hard
end-to-end latency. Setup, workspace review, persistence, client observation and
cleanup have separate scopes; dispatch limits are not guaranteed return deadlines.

Within each task/preset pair, final scoped diffs match, and known reported totals
match where available. Repeats at temperature zero show little variation beyond
timing; they are not independent evidence of broad problem-solving ability.
Matched first inputs/wires differ across arms only by the intended existing
preset instruction/schema; later histories and model choices diverge. The preset
includes its necessary instruction, so tool availability alone is not isolated.

The earlier selected formatter [preset comparison](TOOLSET_RESULTS.md) favored
whole-file completion (2/2 versus full 0/2). Keep that result and every original
cohort unchanged. The present two selected public parser tasks do not establish
a general, causal, held-out or two-model gain, and do not promote any new default.

## Evidence audit and preservation

Every run matches SQLite and native CLI GET restart JSON; archived response/profile
entries match the saved run. Exactly one coding submission/session per cell is
retained, without fork or reversal. All 54 normalized/wire pairs match saved request
identities/allowances and production serialization reconstructed **offline with
MockTransport**. Raw input presence is not backend receipt or generation progress.
Private input hashes and 0700/0600 permissions remain intact; bodies stay local.

Full archived trees, bounded diffs, protected files/modes and original external
grades/regrades match. A further post-execution regrade under the managed Node
binary reproduced all outcomes without artifact changes. Four independent T08
source reviews retain reviewer/verdict/criterion/diff hashes. The final audit
verified **228 outcome-evidence hashes**, the immutable freeze's source/recipe/
preparation hashes, current last artifact and Duodoro head/status/index/refs
fingerprints. This is scoped Git preservation evidence, not a full-byte audit of
Jorge's working checkout or hostile-code isolation.

All **16 owned execution/review APIs** exited, port18760 is released and Gemma is
unloaded. The existing Ollama service remains. No Duodoro working-checkout edit,
commit/push, paid/additional model, dependency installation or Docker action
occurred. Local artifacts, plaintext inputs, recipes and receipts remain at:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-parser-edit-live-xstlu_l9`

Two reporting corrections are retained separately from execution: an initial
read-only audit compared an archived wrapper to its inner response; another
additional post-audit pass inherited a parent Node shim. The final audit compares
the inner response/profile and explicitly pins the frozen managed PATH/Node bytes.
The earlier helper/directory/pass are retained. Original coding and grade/regrade
flows already used the managed environment; no model input, artifact, saved
record or coding attempt changed for these corrections.

## Disposition

The finite comparison is complete. Exact editing activated and preserved behavior
in both selected T08 repeats; it failed to apply either selected T02 patch. Whole-
file refactors again passed a narrow visible check while failing parser preservation.
Keep these task-specific observations and failures visible without turning two
successful refactors into release readiness.

Original ten-task 2/10 results, original 8/10 target, no-go/feature freeze,
production settings and G2–G5 remain unchanged. No broader supported envelope,
model ranking, default change, new inference or release approval follows. These
closed cells are not retry slots. Further development requires a concrete retained-
failure mechanism or an explicitly proposed narrower scope with independent
acceptance. No additional feature or evaluation batch is part of this three-PR slice.

This docs/data report uses the frozen G0 (116 focused/1,453 offline passes, five
live deselected, Ruff/format/mypy and both scripted 90/90 matrices) and the new
retained-outcome audit. It does not rerun the full gate as model-quality evidence.
