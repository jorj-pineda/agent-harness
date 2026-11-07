# Preserve public inputs and stage fresh cells

`verify_public_capture(directory, expected_index_sha256=...)` checks the distinct
public-pilot input index against the caller-selected hash and current reviewed
public plan. It checks private file modes, exact bounded inventories, six-cell
order, original workspace paths, receipts, unchanged fixture bytes/modes, Settings
and environment pins, runtime/dependency report schemas, request identities and
normalized/native input agreement. Pure production-adapter conversion helpers
validate native JSON without constructing a provider or making an HTTP request.
The original captured bytes are retained, rather than regenerated or rewritten.

For an archived bundle, `original_directory` describes its historical root; it
does not need to exist. Archive verification checks selected report bytes and
schemas, not the historical executable or dependency paths. Fresh execution must
reobserve selected runtime/dependency files and establish its other prerequisites.
The selected hash binds bytes; it does not authenticate an approver or certify
that a producer used the promised transport. The separate producer source review
and mandatory-mock contract tests supply that bounded mechanics evidence.

`stage_public_capture(directory, archive, expected_index_sha256=...)` verifies the
capture before creating an exclusive archive container. It moves the whole capture
to `archive/captured`, verifies it at its original workspace roots, then prepares
six fresh unadmitted fixtures at those same paths. This preserves the absolute
paths in the frozen first inputs. The distinct private `staged.json` receipt is
written last, after inventory and fixture checks. `verify_staged_public` checks
the receipt, preserved capture and all six fresh, unconsumed cells before use.

Occupied or overlapping paths refuse. If moving, preparation or receipt writing
fails, partial artifacts remain in place, without a success receipt or automatic
reset/retry. Inspect that evidence and prepare a separately named bundle if needed.
All observations and moves are non-atomic: outside writers can race the checks;
path checks and permissions do not provide isolation or a filesystem lease.

Scripted regressions cover preserved bytes, exact-path production-adapter input
reproduction behind MockTransport, drift refusal before archive allocation,
occupied/overlapping paths, and a retained partial preparation failure. They do
not establish model quality, observed model identity or real backend settlement.
No old held-out/residency format is accepted or relabeled by this API. Public
owned execution, full rehearsal, concrete reviewed freeze, actual parent receipt
and fresh launch gates remain incomplete. Staging grants no inference authority.
