# Parser edit-preset comparison: protocol

**Prepared, not executed.** This finite comparison tests a mechanism suggested by
the [retained parser regressions](PARSER_PRESERVATION_DIAGNOSIS.md): can the existing
exact-edit-capable `full` preset produce more completed, independently acceptable
parser changes than `whole_file`, without unrelated parser rewrites? It uses local
Gemma and the existing allowances, not new runtime behavior or a larger prompt.
The [plan](parser-edit-plan.json) fixes eight rows before actual execution setup.

## Intervention and limits

Compare the existing compact `full` and compact `whole_file` presets at one clean
published runtime. `whole_file` removes `replace_text` and changes exactly the
existing matching edit instruction. All retained schemas remain identical.
This is a preset plus its necessary instruction, not tool removal alone. `full`
still offers `write_file`; it does not force targeted edits. A successful
`replace_text` result on the target is required to report exact-edit activation.
If full mode chooses only whole-file writes, report nonactivation rather than
crediting a targeted-edit mechanism. A no-edit artifact preserving the regex is
not task success.

There is contrary historical evidence: exact-edit escaping/mismatch loops failed
in earlier trials, and the selected formatter preset comparison favored whole-file
editing. The [feedback diagnosis](EDIT_MISMATCH_DIAGNOSIS.md) fixes diagnostic
coverage, not guaranteed recovery. This experiment tests two different selected
public parser tasks; it does not erase those earlier observations. No decoder,
approximate edit, targeted-only toolset, new hint, automatic retry or repair is added.

## Fixed tasks, clients and order

Use original T02 timer normalization and T08 duration refactoring at Duodoro
`5324cabba4ad3c8ad2e1cd1480218cf087d9b741`. Exact requests, full-source fixtures,
visible tests, setup work, reference bytes/modes and external criteria remain
unchanged. Only `server/payloadParsers.js` may change. T08 still needs the original
independent source-review rubric: actual helper reuse for both fields, removed
duplication, preserved behavior and no unrelated edits. Unknown review is not pass.

**Both tasks use the actual CLI in both arms.** Their original ten-task client was
the panel. This deliberate constant client choice isolates the preset comparison
from browser-operation variation; it is not a repeat of the original client
cohort, a CLI/panel comparison or G2 workflow validation. No recovery, fork or
reversal phase is part of these one-turn tasks.

| Order | Task | Cell | Preset |
|---:|---|---|---|
| 1 | T02 | W1 | whole_file |
| 2 | T02 | F1 | full |
| 3 | T02 | F2 | full |
| 4 | T02 | W2 | whole_file |
| 5 | T08 | F1 | full |
| 6 | T08 | W1 | whole_file |
| 7 | T08 | W2 | whole_file |
| 8 | T08 | F2 | full |

Each row gets one fresh session, independent initially absent storage and a fresh
baseline. All eight use one new evaluation-owned **literal workspace path**, with
one stable CLI owner. Actual path/root identities, stores, managed environment,
model identity and hashes belong in the execution freeze, not invented here.
Within each task/preset, repeated initial inputs must match. Across presets for
the same task, initial messages differ only in the intended edit instruction;
tool lists differ only by `replace_text`. Wire options/output allowance match.
Later histories may diverge, so first-input parity is not whole-run equivalence.

## Fixed budgets and evidence

Every cell retains 300 dispatch seconds, 64,000 total reported tokens, 8,192 output
tokens, 24 tools and 12 model iterations. Keep 2,048 per-request output, 256 minimum,
16,384 context, byte/token estimate ratio 3, temperature zero, thinking false,
one completion retry, repeated-call limit two, required configured
`["python", "-m", "pytest", "-q", "test_visible.py"]`, captures on, plan gate off,
root injection/support/inspection off. Provider timeout **180** is held equally
as the existing experimental candidate; it is not adopted as a production default.
The CLI read-inactivity timeout is 600, not a server dispatch-budget extension.

At most eight coding runs reserve 2,400 dispatch seconds, 512,000 total reported
tokens, 65,536 output tokens, 192 tools and 96 iterations. No warm-up, extra phase,
budget transfer, replacement, follow-up coding message or manual patch repair.
Setup/final review/storage/cleanup can exceed dispatch time. An unknown delivery
requires original-key read-only lookup, not a new coding submission.

Use the existing equal private recorder in all cells: twelve requests, 1 MiB per
file, 8 MiB total, directories 0700/files 0600, raw inputs local only. Save normalized
and exact wire inputs before unchanged forwarding. Missing/truncated input or
outcome evidence blocks later dispatch and staging. No failed cell is replaced.
Record in-flight missing usage as unknown, retaining known prefix figures.

After each settled run, stop its owned API and unload only evaluation-loaded
Gemma before artifact audit. Preserve the full tree including ignored entries,
bounded review diffs/explicit omissions, original run/session/budget/progress/check
records, all input pairs, grades/regrades and T08 source review. Fresh GET-only
restart inspection must preserve saved JSON and create no inference. Before
replacing the common path, verify archived tree equality, unchanged criteria and
inputs, full cleanup and no later/external edits. Uncertain ownership blocks
remaining cells. Never reset or use Jorge's Duodoro working checkout as a fixture.
Path checks/process-local admission are not cross-process locks or host isolation.

## Acceptance and interpretation

Primary counts are **completed-and-independently-accepted /4 per preset**, also
**/2 per task/preset**. Report artifact acceptance, completion, observed check
state, known-budget evidence, unrelated parser preservation and operator actions
separately. T08 executable acceptance alone is provisional until source review.
Preserve every blocked, failed, timed-out and unrun row in its fixed denominator;
a partial cohort cannot support an equal-cohort ranking. Never accept a patch
from filename overlap, visible pass, model prose or reference-text resemblance.

Mechanism evidence needs actual successful exact edits and retained parser
preservation, alongside working artifacts. Rejected/stale/malformed calls and
whole-file fallbacks stay visible. More accepted completions with activation
would support this limited candidate hypothesis; equal/worse counts or absent
activation do not establish a remedy. Report all rows and variation either way.
Two repeats on two selected public tasks are not held-out, causal, general or
two-model quality evidence. No default promotion or release-gate closure follows.

## Preparation and remaining gates

Fresh original T02/T08 fixtures fail visible/external checks; their references
pass both. Requests, criteria, setup files/modes and reference hashes match the
original frozen plan. Duodoro head/status/index/refs fingerprints are unchanged.
28 existing criterion regressions and the shared-preset schema/real-write regression
passed. These are offline mechanics checks, not model results or a fresh full G0.

Local preparation evidence and recipe:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-parser-edit-plan-ozbxfk31`

The initial helper redundantly copied its recipe onto itself and raised
`SameFileError` before fixture creation. That root is retained; the corrected
recipe ran in the fresh root above. No scheduled storage or inference was created.

Protocol review/merge precedes the second PR. That execution-preparation PR must
pin actual fresh cell roots/stores, model/environment/runtime/tree/assets/settings,
source/criteria/recipes and full baseline hashes. It must exercise the actual CLI
source factory with scripted providers, real exact/whole-file tools, configured
checks, captures, independent grades and guarded/default GET restart; test first
normalized/wire no-forward mismatches and missing-recording staging refusal.
Run focused regressions, the exact runtime full offline/static gate and both
scripted matrices with unique paths and evidence labels inspected. An independently
reviewed merged execution freeze plus unchanged launch checks is required before
the separate eight-attempt inference/results PR. Failures are retained, not
silently repaired into another freeze after dispatch.

Original cohorts, the 8/10 release target, no-go/feature freeze and G2–G5 remain
unchanged. This new protocol defines a controlled retained-failure hypothesis;
it is not permission to continue any old session or publish a Duodoro patch.
