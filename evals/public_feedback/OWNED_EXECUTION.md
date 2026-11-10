# Owned serial feedback execution

`run_feedback_cohort` binds an admitted eight-cell selection to the existing shared
real-task evaluator and independently owned per-cell backend lifecycle. The caller
must supply an explicit transport factory and scripted/live label. There is no
launch CLI, default network transport, approval generator, retry or resume path.
Actual authority remains the trusted admission caller's responsibility.

Before the single-use schedule marker, the executor rechecks the selected admission,
withdrawal, preserved manifest, clean source, complete Settings, selected runtime/
dependency inputs, archive and staging receipt. All eight staged cells must still
be fresh and match their ordinal, arm and repeat. The original task's initial-fail/
reference-pass controls run before the first backend. These observations are not
a filesystem lock, complete dependency closure or process isolation.

Each cell gets a fresh copied model image and a new owned local backend. The
shared lifecycle verifies image/executable bytes and metadata, exclusive listener/
process-group ownership, and clean settlement; it never manages the user's default
11434 service. Native request/response evidence and the admitted first-input/
Settings/usage/model guards wrap the explicitly supplied transport. Real execution
also records the existing GET-only residency observations; scripted mode omits
that observer. Point observations are not a continuous residency or ownership lease.

The shared evaluator admits exactly the selected arm workspace and grades the final
artifact against the original independent acceptance. The result is saved before
provider/backend cleanup. Runtime configuration is compared to the capture, apart
from observational preservation checkpoint counts. Runtime/provider errors, unknown
usage, model/criterion/preservation drift and unexpected terminal states stop later
dispatch. An independently rejected artifact with known usage and an allowed
terminal state remains a recorded result and does not itself stop the schedule.

A cell becomes recorded only after provider closure, owned backend settlement and
final input rechecks. Cleanup failure can therefore leave a saved result with a
started ledger cell and later cells unrun. Partial directories, marker, admissions
and reports are retained; they are not replacement slots. Setup/image checks,
external grading and cleanup remain outside the per-cell dispatch limit, so the
executor provides no hard end-to-end deadline.

## Validation

Seven focused cases cover active-loop refusal, exact eight-cell arm/repeat binding,
native evidence, clean settlement order and no reentry, transport and cleanup
failure retention, image failure before backend/cell admission, withdrawal before
the marker, and rejected artifacts that can advance. Synthetic tests stub source/
image preflight and backend ownership; positive parent callbacks are scripted
fixtures, not real authority. Existing public-pilot executor tests remain separate.

A separate **native mandatory-mock owned rehearsal** passed from clean source
`9098b634095e141641672702bc9b71dc69217324`. Fresh capture/staging and model-image copies used the actual existing
Ollama executable. All eight cells completed and passed independent acceptance
using scripted reference responses: 24 mock generation requests, zero generation
forwarded to a backend, eight verified clean settlements and no forced cleanup.
Each owned process group was also absent at the final verification point. These
successes demonstrate lifecycle/evaluator mechanics, not lower-tier-model quality.

The [retained summary](owned-proof-20261110.json) binds the capture, manifest,
report and session hashes. Private recipes, inventories, native traces, copied
images and cleanup/session evidence remain at:

`/Users/jorge/.local/share/agent-harness/trials/public-feedback-mock-hc9md743`

The positive parent receipt and callback in this rehearsal were explicitly
synthetic fixtures, not actual approval or evidence of a reviewed execution freeze.
Generation used mandatory in-memory transports; only metadata/ownership/lifecycle
checks used the real local backend. No model downloads, installs, user-service
changes, model inference or paid calls occurred. Eleven selected source dependency
files were pinned, not a complete import/shared-library closure. This proof is
historical to its named code revision; a later head needs fresh matching preparation.
It is not an admissible live freeze or authorization to reuse consumed cells.

Next: fresh retained preparation and candidate-matched rehearsal/probes, concrete
freeze review/CI/merge, actual approval and fresh launch gates before inference.
The original criteria, application defaults, closed cohorts, release no-go and
G2–G5 remain unchanged.
