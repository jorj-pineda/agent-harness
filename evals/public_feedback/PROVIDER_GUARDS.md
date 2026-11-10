# Admitted feedback provider guards

`load_feedback_pins` selects one of eight fixed feedback ordinals. It verifies the
entire selected archive before and after extracting the normalized first request
and its first native request bytes. Invalid ordinals refuse before archive reads;
changed selections refuse without rebinding. It allocates no output or provider.
As with archival verification, it does not reobserve historical host dependencies.

`build_feedback_cell_provider` requires an explicit feedback execution admission,
Settings object, transport and ownership callback. It composes the existing shared
Settings pin, normalized/native first-input guard, native completion checks,
reported model/usage guard and exact owned-endpoint transport boundary. The shared
response guard's type annotation now also accepts the feedback plan; its runtime
behavior and older paths remain unchanged.

Selected source/receipt/withdrawal and preserved manifest bytes, current clean
runtime, complete Settings and selected runtime/dependency inputs are checked.
Ownership is supplied explicitly and must refuse unknown, dead or foreign resources.
These guards bracket transport delegation. Normalized first-input mismatch refuses
before the adapter; wire mismatch refuses before forwarding. Later requests remain
subject to the ongoing source/Settings/dependency/ownership checks.

Construction makes no request and creates no backend or default network transport.
On construction failure the caller owns the supplied transport's cleanup. On
success, closing the returned provider closes the wrapped transport. HTTP refusal,
missing usage, wrong model, missing native completion and observed drift latch
closed instead of retrying or returning tool-bearing responses.

The trusted caller still owns actual approval verification, per-task preservation,
native evidence collection, complete owned lifecycle and settlement. The factory
does not authenticate a receipt, create approval, prove continuous ownership or
provide isolation. It is not a launch API or premium-model supervisor.

## Evidence and next steps

Twenty-six focused tests pass. They cover all eight exact pin selections, invalid
ordinals, mid-extraction drift, ordinal-eight normalized/wire refusal, exact native
forwarding, pre/post-delegation drift, withdrawal, HTTP refusal and invalid native
tool responses. Tests use mandatory mocked transports with real serialization and
real selected artifact checks; source/image preflight seams and parent assertions
are explicitly scripted fixtures, not actual authority.

A separate read-only extraction from the retained native-preflight archive at
`cfa4cddf87585b6421dcd8f749c60e4224be80acc40c8072c5e9ad22bbda058a`
returned all eight exact first-input pairs without provider/transport construction.
That historical capture does not match the current execution source head and was
not admitted or forwarded. No inference, backend or actual approval occurred.

Next: the fixed eight-cell schedule/ledger, then explicit owned execution integration.
Fresh retained preparation, full owned rehearsal, concrete freeze review/CI/merge,
actual approval and fresh launch gates remain before live requests. Original
criteria/defaults/cohorts, release no-go and G2–G5 remain unchanged.
