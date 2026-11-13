# Fresh owned mandatory-mock rehearsal

`rehearse_fresh_owned` is a synchronous internal rehearsal API. It creates only a
new consumed bundle, selects an existing external Python and clean retained source,
records runtime/dependency observations, and runs all twelve selected fresh cells.
It issues no actual or synthetic inference receipt, admits no live candidate and
has no live-generation or injectable transport mode.

The existing capture driver is split into private preparation/cell/finalization
steps so both entry points reuse identical production-adapter mock responses,
normalized/native evidence, shared task evaluation and preservation checks. The
original asynchronous `rehearse_fresh_inputs` contract is unchanged. Each cell still
uses three internal write/check/final requests with canned usage, never a backend
model request. Callback ownership checks bracket every mocked adapter request.

Each cell gets its own verified APFS model image via the existing `prepare_image`
helper, then a separately owned metadata-only backend through `owned_backend`.
The existing lifecycle starts outside an event loop; the shared capture coroutine
runs with `asyncio.run` only while that backend is owned. No await timeout or worker
thread is used to abandon a subprocess. Shared ownership code manages cancellation
and cleanup and never manages the user's default 11434 service. The selected owned
port remains 11447; readiness uses bounded GET-only metadata observations.

A result is retained before backend settlement. No next cell advances until clean,
unforced, exit-zero cleanup is recorded and agrees with the retained session.
Earlier image/session/cleanup hashes are rechecked before later dispatch and final
publication. Unknown/forced/failed settlement, source/dependency/runtime drift,
image failure or corrupted retained evidence stops the rehearsal. Partial resources,
results and failure markers remain; there is no reset, retry or resume. The capture
success index is finalized only after all twelve settlements, then the owned success
summary is written last. If later owned-summary publication fails, an already
completed capture subphase can remain alongside an owned failure marker; it is not
an owned-success or inference-approval record.

Checks cover the selected executable and dependency-file observations, not complete
import/shared-library closure or authenticated identities. Independent copies, path
checks and process-group ownership are not host process isolation or continuous
writer attribution. Native generation stays mocked even though lifecycle metadata
uses real HTTP. Server logs and backend home keys are not evidence inputs to this API.

Focused tests retain explicit fake image/backend seams. They exercise all twelve
serial open/result/settle sequences, ownership checks, original reference acceptance,
no live HTTP in those fixtures, occupied paths, event-loop/overlap refusals, failed
ownership, forced/failed/unknown settlement, prior-record corruption and image
failure. Capture serialization/cancellation tests rerun as compatibility checks.
Those tests establish orchestration mechanics; actual native image/backend proof
must be retained separately without pretending these seams prove real settlement.

Next: retain a clean-source native owned rehearsal and candidate preparation/gates,
then bind a concrete execution freeze, actual approval and fresh launch checks.
Public authored tasks stay development evidence. No new inference authorization,
quality/default adoption or release-gate closure follows from this rehearsal.
