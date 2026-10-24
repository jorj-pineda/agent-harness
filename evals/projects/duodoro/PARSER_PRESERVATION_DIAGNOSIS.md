# Retained parser regressions: visible pass, external rejection

Unchanged T02 and T08 artifacts from the [provider-timeout cohort](PROVIDER_WAIT_TEN_TASK_RESULTS.md)
were copied into fresh disposable directories. Their original visible tests pass;
their original external acceptance and regrade both reject. Three direct Node
counterexamples reproduce the same unrelated `parseSetIntention` regression in
each artifact. No model output was repaired and no test or criterion was changed.
[Structured evidence](../../results/2026-10-24-duodoro-parser-preservation-diagnosis.json)
retains the probes, hashes, existing source review and recipe identity.

| Input text | Required original behavior | Both retained artifacts |
|---|---|---|
| Leading/trailing spaces and three spaces between words | `work well` | `work   well` |
| Tab then newline between words | `work well` | Tab/newline remain between words |
| Literal backslash followed by `s` between words | Preserve those literal characters | Replaces them with a space |

Both whole-file rewrites changed `.replace(/\s+/g, ' ')` to
`.replace(/\\s+/g, ' ')`. The latter matches a literal backslash and one or more
`s` characters instead of whitespace. This changes behavior outside the requested
duration normalization/refactor even though the path itself was allowed. File
allowlists and unchanged protected files do not establish semantic preservation
inside an edited file.

The frozen T02 visible test covers fractional timer values and an infinity clamp.
It does not call `parseSetIntention`; the external test already checks repeated
spaces and rejects this artifact. The frozen T08 visible check calls the exported
helper once. Its external golden behavior checks already cover other parsers, and
the original independent source review records **fail** for parser preservation,
despite finding actual helper reuse and removal of duplicate duration expressions.
That review is retained, not replaced with a new passing verdict.

T02 originally stopped `budget_exhausted` / `not_run`. Its passing check here is a
new **offline artifact observation**, not a claim that the model checked or finished
the task. T08 originally completed with an observed passing check; those fields
are unchanged. They report the loop/check observations, not independent acceptance
or the truth of every answer claim. Neither artifact becomes accepted.

The copied trees match their retained source before and after testing, including
bytes, modes, directories and ignored entries: 454 entries for T02, 464 for T08.
All 360 original outcome-evidence hashes match before and after the recipe. Node
reports **v24.22.0** with the frozen binary hash. The visible pytest invocation
adds `-p no:cacheprovider` and disables bytecode writes for these read-only probes;
test contents/assertions are unchanged. Original configured check argv and saved
observations are not rewritten. Independent acceptance uses the frozen runtime's
owned grader; these trusted local subprocesses are not process isolation.

Local recipe, copied artifacts and complete check output:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-retained-failures-k5qia1i5`

All 28 existing Duodoro criterion regressions passed. A separate read-only hash
check also matched all 376 frozen preparation-evidence files and 25 recipes. This
docs/data slice changes no application/evaluator behavior; it does not rerun the
full gate or scripted matrices. No inference, service, dependency installation,
Duodoro checkout edit, Docker action or patch repair occurred.

The existing independent boundary correctly rejected both changes. This evidence
does not establish why Gemma changed the regex, that a prompt reminder would avoid
it, or that targeted editing would succeed within the same budgets. Keep the
original counts, criteria, no-go and open G2–G5. A future controlled edit-mechanism
experiment would need its own agreed inputs and independent acceptance; these
closed failures are not retry slots.
