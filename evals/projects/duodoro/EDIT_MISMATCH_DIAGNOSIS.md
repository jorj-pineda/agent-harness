# Exact-edit mismatch diagnosis — 2026-10-19

This is **offline replay of recorded tool arguments**, not another model cohort.
The frozen [trial results](TRIAL_RESULTS.md), acceptance criteria, source fixtures
and execution manifest remain unchanged. The original 2/10 artifact acceptance
and 1/10 completed-and-accepted results still stand.

## Reproduction and cause

The saved run responses contain nine dispatched `replace_text` failures across
T01, T04 and T09. Each recorded hash matches the corresponding unchanged fixture.
Replaying each argument dictionary through the real registry against a temporary
copy reproduces rejection without changing bytes or permission bits.

| Task | Dispatched mismatches | Why the prior specific hint did not fire |
|---|---:|---|
| T01 | 2 | Both spans mix literal backslash-n with a real final newline. The second also escapes quotes. |
| T04 | 4 | Newline conversion alone still leaves backslash-escaped quotes unmatched. |
| T09 | 3 | Newline conversion alone still leaves backslash-escaped quotes unmatched. |

The later T04/T09 batch records labeled "Tool not executed" are not dispatched
mismatches and are excluded from this nine-call replay. None of these original
calls applied a patch. T04/T09 repeated some argument dictionaries; this is nine
recorded calls, not nine independent task samples.

The old diagnostic required no real newline anywhere in `old_text`, and only
considered replacing literal backslash-n. It therefore returned generic copying
advice for all nine calls. Removing that restriction and considering escaped
double quotes, separately or with newlines, yields a unique diagnostic candidate
for each of the nine original spans.

The retained evidence consists of normalized tool records, not original Ollama
wire captures. It does not establish where upstream extra escaping originated.
Inspection shows the Ollama adapter passes parsed argument dictionaries through;
mock HTTP response regressions confirm that these escaping shapes survive JSON,
provider normalization and registry validation unchanged. No adapter decoding
defect is demonstrated or repaired here.

## Focused change and boundaries

`replace_text` now explains these escaping patterns only after exact matching
failed and a diagnostic candidate occurs exactly once in the current file. The
message asks for exact `old_text` from `read_file` and inspection of `new_text`
for the same extra JSON escaping. It explicitly reports rejection and unchanged
arguments. It includes no file excerpts or expanded replacement payload.

Candidate strings are diagnostic only. The editor still requires the supplied
exact span, unique match and current whole-file hash, then performs its existing
bounded content/mode recheck before publication. Exact literal escapes retain
priority, including when an interpreted alternative also exists. Ambiguous
interpreted candidates, unrelated text, tabs and CRLF mismatches retain generic
advice rather than a guessed match. No general escape decoder, automatic edit,
new retry, recovery prompt, schema change or budget change was added.

The new synthetic wire/tool regressions cover mixed and fully escaped newlines,
escaped quotes, explicit caller correction, preserved arguments/bytes/modes,
literal source escapes, ambiguous candidates and stale-hash rejection. They are
scripted mechanics evidence, not proof that Gemma will correct its next call.
The earlier clearer-errors comparison also failed to demonstrate an acceptance
gain; see [the historical baseline](../../BASELINES.md#2026-10-04--gemma412b-clearer-tool-errors-thinking-off-context-4096).

## Replay evidence and next measurement

The local replay report and recipe are retained separately from the frozen trial:

`~/.local/share/agent-harness/diagnostics/exact-edit-feedback-20261019-8va4wc5_/`

The recipe verifies the recorded source hashes, invokes all nine calls against
fresh temporary files, and asserts rejection with specific feedback plus unchanged
bytes, modes and arguments. It records run/source hashes and old/new errors, not
source contents or repaired patches. From a checkout containing this fix, replay
with its Python environment and `PYTHONPATH` set to the checkout; the recipe needs
the original local trial bundle at its documented path. Synthetic regressions are
portable; the full local replay is not a bundled public-source fixture.

All nine replays receive specific diagnostics and remain rejected. This identifies
a feedback coverage defect; it does not show improved model task completion.
Provider-error reporting and a separately frozen same-task/model comparison remain
next work. No inference, paid calls, Docker actions or Duodoro working-checkout
edits occurred during this investigation. Release gates remain unmet.


Validation: 90 focused tests passed (two live tests deselected); the final wording
also passed 62 focused file/feedback regressions. The full offline gate passed
1,304 tests with five live deselected. Ruff, all 254 Python files' formatting and
mypy passed. Both scripted matrices passed 90/90 escalation decisions at unique
temporary paths; scripted-contract labels and metric limits were inspected.
The pre-existing Chroma deprecation warning remains. No live-model recovery or
acceptance improvement was measured.
