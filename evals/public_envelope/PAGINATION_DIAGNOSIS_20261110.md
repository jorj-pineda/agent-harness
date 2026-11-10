# Pagination failure: explicit requirement, passing visible check, rejected artifact

Both rejected feature artifacts from the [closed public pilot](RESULTS_20261109.md)
reproduce through real file/check tools on disposable fixture copies. Their
visible checks pass; unchanged original independent acceptance reports **2 failed,
19 passed** per artifact. No inference, original-artifact repair or criterion
change occurred. [Structured evidence](pagination-diagnosis-20261110.json) binds
the local recipe, runtime/tool/grader sources and source report.

## Requirement and check boundary

The normalized first user input in both cells explicitly says `bool is invalid`.
Both final implementations nevertheless use `isinstance(value, int)`, which admits
Python bool values. A subsequent positivity check rejects `False` but admits `True`.
The requirement was supplied; its presence does not explain why the model missed it.

Direct copied-artifact probes cover both argument positions and empty/nonempty
input independently:

| Argument under test | Input | Both retained artifacts | Original requirement |
|---|---|---|---|
| page | True | returns a page, including [] for empty input | ValueError |
| page_size | True | returns a page, including [] for empty input | ValueError |
| page | False | ValueError | ValueError |
| page_size | False | ValueError | ValueError |

Each artifact has eight probes: four unexpected returns for True and four expected
ValueErrors for False. The unchanged reference rejects all eight invalid inputs,
passes the visible check and passes all 21 original acceptance cases. This is a
reference/control observation, not a repaired model artifact or new model success.

The visible test contains two cases: ordinary second-page slicing and preserved
first_page behavior. Neither exercises boolean validation. The existing independent
check does, and correctly rejects both artifacts. In its True cases the first
`pytest.raises` fails before reaching the second argument's assertion; the separate
diagnostic probes above verify both positions without changing that acceptance file.

This localizes the observed gap to requirement implementation and visible-check
coverage. It does not identify the model's internal cause or establish that exposing
more cases would make it repair the implementation within the same budget.

## Duplicate write and stale verification

Both saved tool sequences are:

`tree → read_file → read_file → write_file → run_project_check → write_file`

The two writes in each cell carry identical full-file arguments. Fresh real-tool
replays confirm identical content hashes after both writes and equality with the
retained final implementation. The writer still performs its documented atomic
replacement; the replay observes a different inode after the second write.

The outcome harvester invalidates earlier check evidence after a successful file
write. Thus the final check is stale despite identical source bytes. This is the
current conservative contract, not evidence of a failed subprocess or lost check.
The six-iteration limit stops after that write without another model response.
The historical completion/verification states remain budget_exhausted/stale.

Avoiding this duplicate mutation or retaining check evidence for a proven no-op
could save redundant work in a future design. Neither would fix the independent
bool rejection in these artifacts. This diagnosis justifies no verification-rule
change, larger limit, automatic retry or successful-completion reclassification.

## Reproduction and preservation

Reporting source: `b76ca2c319a06be09dd3347a09e526e2bf567ff7`. The diagnostic replays
the two saved six-tool argument sequences through the current shared registry,
whole_file_check tool selection, real atomic writes and configured check. It does
not replay inference, loop scheduling or latency; outcome harvesting is evaluated
on the newly executed tool records. The relevant source hashes are in the evidence.

All workspaces and memory databases are temporary. The check environment uses an
explicit Python path, disables automatic pytest plugin loading and bytecode writes,
and does not inherit provider credentials. Existing tests and acceptance are copied
or read unchanged. Direct probes import only trusted copied fixture code. These
are trusted-local subprocesses, not OS isolation.

All **159** selected original evidence hashes match before and after the diagnosis.
The local recipe and its output/log are retained separately:

`/Users/jorge/.local/share/agent-harness/trials/public-pagination-diagnosis-9fkahjg6`

Run from the project checkout with its existing environment:

```sh
uv run python /Users/jorge/.local/share/agent-harness/trials/public-pagination-diagnosis-9fkahjg6/recipe.py
```

The recipe reads the private retained pilot, creates disposable directories and
writes its summary under `/tmp`; it makes no provider requests. Recipe hashes bind
observed bytes, not authenticated provenance. Timing strings in regrade summaries
can vary between runs; counts, exit states and behavioral outcomes are the claims.

Focused existing outcome/refactor/category regressions passed **38 tests**. This
docs/data slice changes no application/evaluator behavior and does not rerun the
full offline gate or scripted matrices locally. No model service, dependency,
production source, original fixture, acceptance criterion or frozen plan changed.

## Next experiment boundary

The requirement/check gap is now concrete. A future **fresh public development**
comparison could test a visible check covering the already stated bool restriction
against the existing visible check, with the same model, requests, independent
acceptance and comparable budgets. Use separate fresh fixtures and a reviewed
protocol/input freeze; never replace the old frozen test, resubmit closed cells or
expose private held-out criteria. Such a comparison would test feedback usefulness,
not establish general feature quality or a release envelope by itself.

This diagnosis selects no treatment/default and authorizes no inference. The
original 4/6 public results, 2/10 latest project results, 8/10 target, release no-go
and open G2–G5 remain unchanged.
