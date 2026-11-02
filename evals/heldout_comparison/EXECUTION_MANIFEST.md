# Execution manifest and parent-receipt binding

The read-only load_execution API takes a manifest path/hash and separate parent
approval path/hash. It bounds private regular files, validates strict typed schemas,
requires the parent receipt to bind the exact manifest hash and runtime revision,
then rechecks both artifacts. It creates no directories, providers, services,
requests, receipts or approvals. There is no CLI or approval generator.

The manifest explicitly selects the runtime commit, protocol/capture/runtime/
dependency/image-receipt/binary hashes, staged/capture/output/image/binary paths,
and effective command environment. Paths must be absolute and known bounded
runtime environment keys must include the forced pytest startup protections.
The manifest itself always records inference_authorized=false: preparing bytes
is not permission to execute them.

The separate parent receipt names an approval ID, the manifest/runtime binding,
freeze reviewed head and merge revision, and affirmative review/CI/merge/launch
assertions. Those fields are assertions from the parent, not facts authenticated
by this schema. Hashes select bytes; they do not authenticate an issuer or prove
Git ancestry, review results, CI, merge state, user consent or freshness. A trusted
operator must obtain the receipt from the actual authorized parent and verify
those prerequisites independently. An arbitrary JSON file with true fields must
never become self-approval. Tests use clearly scripted receipts only.

This is a binding check, not complete launch admission. Future runner code must
still verify the actual clean retained runtime and selected artifacts, source/
criteria and full sixteen-cell schedule, fresh unconsumed prepared workspaces,
nonoverlapping canonical output paths, current model image/version and owned
resource startup, request guards, single-use output reservation, final preservation
and owned settlement. Failure or unknown evidence stops without rebinding/retry.
The live runner and concrete exact freeze are still incomplete; no inference or
launch authority follows from loading this tuple.

The separate `evals.heldout_execution_preflight.preflight(manifest)` API now
checks semantic artifact bindings read-only, including the actual clean runtime,
current protocol, capture settings/environment/runtime/dependency selections,
all sixteen fresh staged fixtures, prepared model receipt/image, and selected
executable bytes and executable mode. Managed directories must be canonical,
outside Git workspaces and pairwise nonoverlapping; the output must be absent.
Aliases, an occupied output, and an executable inside managed directories refuse
before large artifact inspection. Runtime and paths are rechecked at completion.

This API is usable during freeze preparation without an approval receipt. It
creates no output, admission marker, provider, service or request, and supplies
no approval. Image verification reads the full selected image and may take time.
Checks are bounded and non-atomic: they do not lock external writers, prove full
dependency closure or reserve paths/ports. Current service version, exclusive
single-use execution admission, request-time checks, provenance, and settlement
still belong to the future trusted live operator. Passing this preflight alone
cannot start or authorize inference.

`evals.heldout_execution_admission.admit_execution` provides a separate single-use
reservation step. It requires both selected files/hashes and an explicit trusted
`verify_parent(manifest, approval)` callback with no default. The callback must
raise unless the authorized parent and real exact-head review/CI/merge/launch
prerequisites have been independently verified. This is a trust boundary for
operator code, not authentication implemented by this package. A no-op callback
does not establish authority. Scripted tests use synthetic callbacks and receipts;
no actual launch approval has been issued.

After binding and provenance checks, admission runs semantic preflight, rereads
the selected private bytes, and exclusively creates the fixed output directory.
The exact manifest and receipt are preserved as private files and verified before
`admission.json` is written last. The record says admitted, not executed or
completed. Source selections must be canonical, outside managed trees and Git.
Concurrent attempts with the same fixed output can reserve it only once. Any
post-allocation failure retains partial artifacts; there is no rollback, deletion,
reset, output reassignment or automatic retry. Failure-recording errors do not
replace the primary failure. Caller callbacks receive deep copies of the models.

This function starts no provider, tool, service or inference and admits no task
workspace. It is not the cohort runner or a full execution freeze. Future trusted
runner code must use the admitted selections, fresh request/resource gates and
the owned lifecycle, retain every planned result/unrun cell, and stop on uncertain
submission or failed cleanup. The reservation does not lock external writers or
provide power-loss durability, isolation, continuous monitoring or settlement.

The admitted-cell factory in `heldout_execution_provider` composes the production
Ollama adapter with first normalized/wire input pins, actual Settings, native
model/completion/usage checks and the exact owned endpoint transport. It requires
an explicit transport and trusted ownership callback; it creates neither. The
callback and preserved manifest/runtime/dependency checks run before construction
and around HTTP delegation. Settings drift also refuses before adapter dispatch.
There is no default network connection or launch CLI. Successful factory return
transfers transport closure to the provider; on construction failure the caller
still owns closing its transport. The caller must supply valid admission and
manage lifecycle, task admission, result recording and settlement. This factory
is not an approval authenticator, resource lease or complete cohort runner.

`heldout_execution_cohort.run_admitted_cohort` accepts trusted admission and an
explicit transport factory (without a network default). It requires an explicit
scripted/real evidence label. After a private exclusive start marker and initial
ledger, shared controls run before any backend. Each fixed cell gets a separate
synchronous owned backend context; task execution and provider closure happen in
an event loop inside it. The result row is persisted before backend exit, and
the next cell starts only after successful owned cleanup and preservation checks.
Provider/criteria/usage/runtime/preservation/cleanup failures stop the cohort;
ordinary artifact rejection/native budget outcomes can continue after settlement.
Report failures do not trigger another attempt. A failed final report may leave
the last safe ledger incomplete; secondary recording errors annotate the primary.

The API authenticates no constructed Admission object. Trusted operator code must
use actual approved admission, provide a no-retry transport factory that performs
no submissions itself, and retain all lifecycle evidence. Tests replace ownership
and settlement with scripted contexts and mock responses. No actual freeze or
parent launch approval has been issued, and no generation was run to validate this
slice. Concrete operator/no-forward/refusal probes and fresh launch gates remain
required. Boundary checks are non-atomic, not process isolation or a hard end-to-end
deadline. The generic metadata-only CLI remains metadata-only.

Each cohort transport now writes private native evidence under cell-NN/native.
Requests are saved exclusively before submission; responses are read at most
2,000,000 bytes in bounded chunks and saved before returning to HTTP/native guards.
HTTP errors retain their exact bounded body too. Incomplete read prefixes are
labelled partial; failure records distinguish pre-delegation from uncertain
submitted work and do not include exception text or headers. Authorization/cookie
headers and unexpected routes refuse before request capture. Oversized payloads,
recording/read/closure failures, concurrent use or more than32 submissions latch
closed. The count is above the fixed six-iteration/correction envelope, not a new
allowance to bypass runtime budgets. Encoded responses refuse rather than claiming
decoded bytes are wire bytes. Caller-supplied transports may already buffer bodies;
this wrapper bounds its own reads/copies, not arbitrary delegate allocations.

The recorder sits inside the owned transport: ownership/pin checks precede saving
and delegation and follow the bounded body read. It creates no network transport
and supplies no approval or settlement. Files are exclusive/private but not atomic
as a set, writer-isolated or power-loss durable. A recording failure can retain
partial files and an incomplete ledger; it never becomes a retry slot.

The additional scripted cohort failure probes cover control refusal before any
backend opens, provider-construction transport closure before task admission,
provider-close failure after a retained row, selected-manifest drift after a cell,
cancellation after delegation with uncertain native evidence, and report errors
at initial/row/final writes. They check later cells remain unrun and consumed
start markers cannot become retry slots. A failed final write can leave the last
safe ledger incomplete; the caller receives the primary exception with secondary
recording notes. Active event-loop entry refuses before resources. These tests
use synthetic admission and mocked ownership/settlement and do not replace real
resource refusal probes or authorize generation. No runtime policy changed.

Admission now refuses any `WITHDRAWN.json` entry beside the selected source
manifest, including malformed files, directories, FIFOs or broken symlinks. It
checks before parent verification, after that callback, after semantic preflight
and before writing the final admission record. Marker contents are never opened
or interpreted as a permission override. A withdrawal observed after allocation
retains partial preservation/failure evidence without a success receipt or retry.
This enforces the documented withdrawn trial at admission, not just in prose.

Read-only loading/preflight remain available for forensic inspection. No marker
removal/reset/reapproval API is provided. Checks are non-atomic and do not lock
external writers, authenticate a parent or monitor already admitted execution.
Constructed Admission objects remain a trusted-caller boundary; future real
operator code must also recheck withdrawal before resource/request admission.
No launch approval or inference follows from an absent marker.

The explicit `heldout_execution_operator.execute_selected` API connects actual
selected manifest/receipt admission to the real cohort with `scripted=False` and
a zero-retry, environment-proxy-disabled HTTP transport. It has no CLI, implicit
launch or approval generator. A required trusted parent verifier must independently
establish actual authority and freeze/review/CI/merge/fresh-launch prerequisites;
a no-op callback or arbitrary affirmative JSON is not approval. Calling this API
can generate and is not a read-only preflight. No invocation with real resources
or inference was used to test it.

Admission now retains the canonical selected source manifest path in its object
and record. Real cohort entry refuses missing source provenance. When present,
the cohort and owned request guards recheck source bytes/mode and withdrawal at
construction, cell boundaries and around transport delegation/body read. A changed
canonical parent refuses rather than rebinding. Existing synthetic scripted
fixtures may omit provenance but cannot use that omission for the real cohort.
These checks remain non-atomic observations, not issuer authentication or a lease.
The caller's actual parent verifier and complete reviewed freeze remain required.

`heldout_preparation.prepare_candidate` is a one-shot offline entry point for a
new trial, invoked with the selected existing absolute Python from a clean retained
source tree. It requires the expected lexical interpreter path and clean commit,
rejecting a source-local interpreter/prefix or any source-local `.venv` entry; resolving the base Python alone would
not distinguish different venvs. It snapshots explicit executable/dependency/env
selections before allocation, uses the existing clone/capture/staging helpers,
performs semantic preflight, and writes candidate manifest/preparation evidence
last. There is no resolver, installer, service, generation, CLI or parent receipt.

Post-allocation failures retain partial evidence and a withdrawal marker; occupied
outputs refuse without reset or retry. A preparation receipt does not authorize
admission or make dependency/import closure complete. The caller still selects
trusted existing dependencies and exact command environment. Checks are bounded
and non-atomic; this function cannot prevent a launcher from installing something
before it starts or detect every unrelated host mutation. The approved recipe must
use the original absolute interpreter directly, never uv environment resolution.
The earlier withdrawn candidate remains excluded and cannot be repaired into a
launch by this entry point.
