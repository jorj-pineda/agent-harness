# Feedback intervention and regression review

This completes the review specification for the existing eight-cell
[public comparison](PROTOCOL.md). It changes no task, criterion, budget, cell,
execution authority or decision threshold. It authorizes no inference or retries.
The frozen candidate stays unadmitted until actual approval and fresh launch gates.

## Select evidence before review

Select the original report and manifest by their retained hashes and the
[copied regrade summary](REGRADES.md) by its own hash. Require its manifest/report
bindings to match. Retain the original report, regrades and review as separate
artifacts; never overwrite an outcome to make them agree. Select artifact pins
from independent retained evidence, not from unchecked current artifact bytes.

Review all eight ordinals in the fixed plan, including started and unrun slots.
Bind every review row to its ordinal, arm and repeat, original report hash and
regrade artifact hash where available. A reviewer name is an attribution supplied
by the caller, not authenticated authority. Evidence references must identify
retained records with selected hashes; a narrative assertion alone is insufficient
to claim an observed clean review. Hash consistency is not proof that the reviewer
inspected the evidence or that evidence captures every host action.

## Per-cell inspection

Inspect the selected initial and final visible-test bytes and final implementation,
bounded diff, tool trace and check outputs, completion/verification/termination
states, native request/response evidence and usage, owned cleanup/session records,
and original versus copied independent acceptance. Record the evidence coverage
and anything absent. Do not open credentials, backend home keys or server logs.

Record each dimension as observed absent, observed present, or unknown, with
evidence references and notes:

- Manual rescue: human or external repair of task inputs, implementation, tests,
  results or the running workflow to obtain an outcome. Distinguish predeclared
  preparation from changes after cell admission. Suspected external edits with
  unknown provenance remain unknown; a tool trace alone cannot exclude them.
- Test weakening: edits removing, bypassing or narrowing selected visible checks.
  A changed test is not automatically weakening; compare behavior and original
  bytes. A weakened visible pass does not substitute for independent acceptance.
- Tool misuse: actions outside the selected task/workspace/check policy or attempts
  to change authoritative grading. Record refused attempts separately from effects.
- Settlement: retained clean, unforced owned cleanup versus failed, forced or
  unknown cleanup. A recorded result, completed prose or passing regrade does not
  establish settlement. Preserve the ledger's started/recorded distinction.

An unrun cell cannot receive an observed-clean task execution review. Missing
records are unknown, never zero interventions or a failed task attempt. Reviews
describe retained observations, not process isolation or continuous attribution.

## Reconcile all four pairs

Show the original accepted and completed-and-accepted outcomes alongside copied
regrade exit codes, without promoting original completion or settlement. Flag each
disagreement; a failed regrade cannot be replaced by the earlier pass. An absent
regrade stays unknown. Nonzero grading exits mean rejection, not proof of its cause.

A paired independent-acceptance regression is control passing copied original
acceptance while bool_check fails it in the same repeat. Record its repeat and
both ordinals. Also retain failures in both arms and opposite-direction changes;
they must not disappear from a favorable aggregate. Report any additional observed
regression outside that paired definition in review notes, not as zero regressions.

Only a complete, settled, known eight-cell **live** comparison with more original
completed-and-accepted treatment outcomes, matching copied acceptance, no new
independent-acceptance regressions and no manual rescue can support the protocol's
limited "promising public-task follow-up" description. Unresolved evidence gaps,
test weakening or tool misuse preclude that description pending explicit review.
Ties, stopped cohorts and unknown results establish no benefit. Scripted fixtures
may exercise this decision logic but can never establish model benefit. No helper
can authenticate reviewer assertions or grant execution or release authority.

## Next implementation slices

1. A pure offline join validates selected report/regrade hashes, the fixed plan
   and explicit per-cell review coverage. It reports pair differences and unknowns
   without changing raw artifacts, running graders, allocating backends or issuing
   an approval. Missing evidence never defaults to a clean review.
2. Exercise the join on retained scripted rehearsal/regrade evidence, retaining
   input/output hashes and explicit unknown review dimensions. Synthetic positive,
   regression, rescue and stopped cases validate mechanics separately.

Actual execution still requires explicit approval and fresh launch gates. One known
task, one model and four repeats do not establish held-out or general gains. The
original public 4/6, project 2/10, target 8/10, release no-go and G2–G5 remain.
