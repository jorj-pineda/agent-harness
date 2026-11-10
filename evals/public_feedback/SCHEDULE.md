# Fixed serial feedback schedule

`evals.public_feedback_runner.run_feedback_schedule` executes the eight selected
cells in their fixed ordinal/arm/repeat order. Its distinct `FeedbackLedger`
retains every cell as unrun, started or recorded, plus overall planned/running/
finished/stopped state. This is an orchestration seam, not an execution or approval
API, and it constructs no provider or backend.

An exclusive private marker is written before any callback. Its presence prevents
reusing that marker, including after partial failure. The initial ledger is
persisted, and each started state must persist before dispatch. A cell becomes
recorded only after the trusted `execute_and_settle` callback returns. The callback
must have saved its result and settled its owned resources before returning; it
must raise on protocol stop conditions. A return and ledger entry cannot themselves
prove settlement or approval.

Callback, cancellation or recording failure stops later dispatch. The current cell
and later unrun slots remain in the in-memory stopped ledger, and stopped-state
persistence is attempted. A secondary recording failure is attached to the primary
exception instead of replacing it. Neither marker nor partial evidence is removed.
There is no resume, retry, cell replacement or budget transfer.

Callback cells and persistence snapshots are deep copies so external mutations
cannot change the internal schedule. The caller still owns safe evaluation-owned
paths, report persistence, genuine admission, ongoing input/resource guards and
actual execution/settlement. This does not lock out external filesystem writers
or isolate processes; competing calls are excluded only for the same marker.

## Validation and remaining boundary

Eleven new scheduler tests pass alongside nine unchanged public-pilot scheduler
tests. They check the balanced eight-cell order, all state transitions, single-use
markers, cell failures at early/late positions, recording before dispatch, primary
error preservation, callback mutation, protocol stop labeling, cancellation and
competing same-marker attempts. These scripted callbacks do no model work and
establish neither coding success nor backend settlement.

Next is the owned executor callback and shared report integration. Fresh retained
preparation, full owned rehearsal, concrete freeze review/CI/merge, actual approval
and fresh launch gates remain before live inference. Original criteria/defaults/
closed cohorts, release no-go and G2–G5 remain unchanged.
