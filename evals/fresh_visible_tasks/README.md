# Fresh visible-check development tasks

This explicit catalog extends the known pagination result with three newly authored
coding tasks. It uses the existing `task-catalog-v1` loader and shared external
acceptance grader. It does not add tasks to the default catalog or change runtime,
prompt, tools, grading policy or any historical task. It authorizes no inference.

These are **public development tasks**, visible to authors and reviewers, not
held-out or contamination-attested evidence. No target-model execution is recorded
for these task IDs at authoring. New task names alone cannot establish novelty to a
model or independence from training data; similar concepts occur in earlier tasks.
This checks transfer across these particular fixtures, not generalization at large.

| Task | Category | Boundary tested by stronger visible checks |
|---|---|---|
| `fv_lookup_option` | bugfix | A present falsey value must not become the default |
| `fv_merge_windows` | feature | Touching/nested ranges merge without input mutation |
| `fv_header_pair` | refactor | Immutable tuple record preserves first-colon parsing and errors |

Each task has an initial fixture, weak visible test, a separate stronger replacement
`test_visible.py`, a reference implementation and external acceptance tests. The
request specifies all tested requirements in both arms. Stronger tests retain every
original visible test and add boundary examples. Both arms must use identical
initial implementation/request/independent criteria and the same selected runtime,
settings and budgets. Only initial visible tests differ; availability and feedback
remain bundled. No test-feedback recovery mechanism is claimed.

External tests use different examples and broader original requirements than the
visible tests. The reference passes weak, stronger and external checks. Initial
fixtures fail weak and independent checks, as required by shared `validate_task`.
Explicit partial patches pass the weak test but fail both stronger and independent
checks: `get(...) or default`, merging strictly overlapping rather than touching
ranges, and splitting a header on every colon. Replacing visible tests with a
trivial pass still cannot change independent acceptance. These controls demonstrate
test discrimination and grader mechanics, not probable model success.

Validate with no provider or inference:

```sh
uv run python -m evals.real_run --task-manifest evals/fresh_visible_tasks/v1/tasks.yaml --validate-only
uv run pytest tests/evals/test_fresh_visible_tasks.py -q
```

Fixtures run trusted Python on disposable copies. External pytest startup controls
and path allowlists are not host process isolation. Preserve source/criteria and
partial failure evidence, and keep credentials out of future model traces.

## Proposed bounded comparison

The next comparison proposal uses the same pinned local Gemma4:12b/Ollama selection
as the closed pagination comparison, direct/whole_file_check and its per-cell
limits: 150 dispatch seconds, 32K total/4096 output tokens, six iterations, 12 tools,
request output cap 2048/minimum 256 and two touched files. Temperature zero,
thinking off, context 16384, keep-alive 300 and the fixed
`python -m pytest -q test_visible.py` check remain proposed. No warmups, rescue,
retries, replacements, budget transfer or model switching. This document does not
implement enforcement or attest model availability. No paid call is selected.

Use two paired repeats per task, reversing order for each task on its second
repeat. Fixed proposed serial order (weak and stronger are both harness mode, not
the evaluator's minimal-prompt baseline):

| Ordinal | Task | Repeat | Visible arm |
|---|---|---|---|
| 1 | fv_lookup_option | 1 | weak |
| 2 | fv_lookup_option | 1 | stronger |
| 3 | fv_merge_windows | 1 | stronger |
| 4 | fv_merge_windows | 1 | weak |
| 5 | fv_header_pair | 1 | weak |
| 6 | fv_header_pair | 1 | stronger |
| 7 | fv_lookup_option | 2 | stronger |
| 8 | fv_lookup_option | 2 | weak |
| 9 | fv_merge_windows | 2 | weak |
| 10 | fv_merge_windows | 2 | stronger |
| 11 | fv_header_pair | 2 | stronger |
| 12 | fv_header_pair | 2 | weak |

Aggregate reservations would be 1800 dispatch seconds, 384K total/49,152 output
tokens, 144 tools and 72 iterations. They are not expected usage or end-to-end
bounds: setup, hashing, grading, preservation and settlement are outside dispatch.
No cell can borrow another's budget. Fresh initial workspaces/images and owned
backends must be independent per cell. Never use the user's repository as a fixture
or manage their default model service.

All twelve cells must be reported, including stopped/started/unrun outcomes. Show
accepted and completed-and-accepted counts per task/arm and all six paired outcomes;
visible passes remain separate. No completed/verified prose, file overlap or token
similarity substitutes for copied original-criteria acceptance. Known usage totals,
scoped timing, native requests/responses, final diffs, interventions, test changes,
tool misuse and clean unforced settlement must be retained and reviewed.

Only consider a promising **fresh-public-task follow-up** if completed-and-accepted
stronger totals exceed weak totals, no task has a lower stronger count, no pair has
a weak-pass/stronger-fail copied acceptance regression, every result/usage/settlement
is known, copied grades agree and no rescue or new regression is observed. Report
all task-level variation even when the aggregate improves. Ties, stopped or unknown
outcomes establish no benefit. This proposed rule selects no default and does not
replace the original project 8/10 target or close any release gate.

## Remaining work before execution

1. Review and merge this authored task/criterion selection. Preserve reviewed byte
   pins before any target-model access; new criteria after inference require a
   separately labelled task version, never a regrade of the old cohort.
2. The [pinned shared-evaluator workspace seam](WORKSPACES.md) is implemented for
   these task/arm fixtures without changing the pagination-specific executor.
   [Mandatory-mock capture](CAPTURE.md) and [owned lifecycle rehearsal](OWNED_REHEARSAL.md)
   are implemented. [Retained clean-source proof](RETAINED_PREPARATION_20261113.md)
   verifies native lifecycle, the offline gate and separate unused fixtures/images.
3. [Capture preservation/exact-path staging](STAGING.md) is implemented. Bind a
   distinct execution manifest/admission and guarded serial owned provider before
   a concrete clean candidate, settings/dependencies/image/model pins, passing
   exact-candidate offline/install gates, retained evidence and fresh launch checks.
   Obtain explicit approval for that exact comparison before admission/generation.

These are preparation requirements, not authority granted by a YAML label, this
proposal or prior pagination approval. Do not infer improvements from offline
controls. A second intended model, held-out tasks, real-project/client recovery
acceptance and distribution validation remain separate scope decisions. The
original 2/10 project and 4/6 public results, release no-go and G2–G5 stay open.
