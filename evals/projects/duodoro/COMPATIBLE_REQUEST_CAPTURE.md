# Bounded compatible-provider input recording

The evaluation-only recorder can explicitly select `/v1/chat/completions` or
`/inference/go/v1/chat/completions`, or the current Go compatible path
`/inference/go/openai/v1/chat/completions` alongside its unchanged `/api/chat` default.
The selected path must match exactly on a POST, with a JSON object body. Other
paths/methods reject before forwarding. This is developer opt-in instrumentation,
not an API setting, new provider, endpoint discovery or authorization to infer.

Normalized records retain their existing v1 shape and request identity. The
transport writes the original SDK request body bytes before forwarding the same
request object. It does not rewrite messages, tool-call IDs, schemas, options,
headers or URLs. Twelve-request/1MiB-file/8MiB-total ceilings and owner-only
0700/0600 directories/files remain unchanged. Existing paths refuse overwrite;
recording I/O/bound errors poison the recorder and stop onward forwarding.
A provider cannot return proposed tools without a paired wire input.

Transport headers, URL, extensions and response/error bodies are not stored by
this recorder. Message/source content remains plaintext and potentially sensitive;
this is not redaction, secure storage, host isolation or backend receipt attestation.
A failed provider can leave normalized-only evidence; trial orchestration must
retain it, abort that phase and reject incomplete pairing before staging more work.
The recorder is not a general retry/reconnection service.

## Conversation binding and ownership

Both CapturedProvider and FirstRequestGuard preserve the optional provider
conversation capability through pure views. Go's underlying configured provider
still supplies its own User-Agent and per-request stable session header, with zero
SDK retries. The helpers do not mutate global headers or create another SDK client.
The first actual chat claims its recorder/guard phase; different conversation IDs
or mixing bound and unbound calls subsequently reject before forwarding. Creating
a view alone neither starts recording nor claims the phase. Direct legacy calls
remain supported as one unbound phase.

Use one fresh capture/guard per explicit phase and close the original owning wrapper;
views do not own the underlying client. Allocate recording only for actual source
work. A default GET-only restart must not recreate an existing input directory;
use reviewed lazy source allocation and an uncaptured/review-guarded path for saved
inspection. Actual client lifecycle preflight remains necessary for any future
trial. These helpers are not selectable application behavior or safe continuation.

First-input identity/temperature checks and review-only no-forward checks stay on
the original guard, including its forwarding count. A valid bound view cannot
bypass them. Missing/mismatched inputs remain errors rather than rewritten requests.
The fixed two-task probe launcher still has its original scope; this change does
not silently turn it into a hosted-model trial runner.

## Offline validation and limits

Actual OpenAI SDK requests against MockTransport cover the explicit compatible
paths, tool-call/result serialization, pre-forward paired files, private permissions,
stable routing without header retention, pure views, shared guard counts, rejected
mixing, unsupported paths/methods, byte/count/I/O failure, timeout/no retry,
cancellation/body cleanup and owner client closure. A real-file API regression
exercises session binding, one write, provider failure, unknown failed usage and
identical saved GET after restart without another request or capture allocation;
pre-existing bytes/modes and partial work survive.

52 focused and 1,483 offline tests passed (five live deselected), Ruff/format/mypy
passed, and both scripted matrices passed 90/90 at unique temporary paths. Their
scripted-contract labels and metric limits were inspected. The existing Chroma
deprecation warning remains. A routine initial unused-test-variable lint finding
was corrected before the passing gate.

No model inference, paid calls, credential changes, Duodoro working-checkout edits
or historical-input/result rewrites ran. Canned SDK traffic establishes recording/
routing mechanics, not hosted access, pricing/billing, model quality or a useful
task envelope. A future comparison needs current authorized model/access/free-price
checks, fresh inputs/stores, independent criteria and reviewed execution preflight/
freeze. Original cohorts, 8/10 target, no-go/defaults and G2–G5 remain unchanged.

## Deferred allocation (2026-10-25)

`RequestCapture(..., lazy=True)` validates the selected path at construction but
creates no directory until the first normalized `begin`. The default remains
eager for existing recipes. Pure binding, GET-only lifecycle startup and owner
close leave absent or existing recording paths untouched. Actual chat still
requires an absent canonical directory: it never reopens, resumes or overwrites
saved evidence. Allocation failure poisons that recorder before inner forwarding.
Delayed parent-link resolution is rechecked; external writer races remain outside
guarantees. The new Go path is explicit, not arbitrary endpoint discovery.

Wrap the lazy captured provider in the existing review-only FirstRequestGuard
when serving saved inspection. Laziness itself is not a read-only permission:
a chat to a fresh directory would record and forward unless an outer guard rejects
it. This helper does not implement a selected-model factory or execution manifest.
The CLI lifecycle regression below covers scripted restart inspection; a future
trial freeze and actual selected-model preflight remain required.

New scripted regressions cover absent/existing startup, binding/close, actual
pre-forward SDK pairs/routing/private permissions, existing-path no-overwrite,
review/first-input rejection before allocation, delayed parent-link drift and wire
without normalized pairing. No inference or account changes ran.

Deferred-allocation validation: 61 focused and 1,492 offline tests passed (five
live deselected), Ruff/format/mypy passed. Both scripted matrices passed 90/90 at
`/tmp/agent-harness-lazy-capture.iaLnyU`, with contract labels/metric limits inspected.
The existing Chroma deprecation warning remains. These are mechanics, not model
quality or actual hosted-client preflight results.

## Scripted CLI lifecycle regression (2026-10-25)

Three canned compatible-SDK cases now drive the real CLI chat entry point through
the default API Components/settings factory with one named provider selected.
CLI HTTP calls are bridged to TestClient. Provider construction is intercepted
to install SDK MockTransport and capture; FactStore and SDK clients remain
application-lifecycle owned. Each case applies one
real file write, then completes with checks not_run, fails on a provider timeout,
or stops incomplete with missing usage. Two normalized/wire pairs match durable
request identities; private routing/header/error markers stay outside input files.
These are synthetic model responses, not Beam calls or independent task acceptance.

The same constructor path reopens a fresh API lifecycle with lazy capture and an
outer review-only guard. CLI `run --json` makes one GET, returns identical saved
run evidence and leaves capture bytes/modes unchanged; saved session JSON is also
identical. Both SDK clients close. Pre-existing protected bytes/modes and a later
user line survive. The timeout retains absent final response and unknown failed
usage; missing usage stays unknown rather than zero. Exactly one selected provider
is constructed per lifecycle, with no model request during review.

This is in-process TestClient lifecycle coverage, not OS-process/browser restart,
a selected Beam execution source factory, hosted connectivity, reasoning-history
compatibility, cancellation preflight or a reviewed execution freeze. Those remain
separate. The first test draft used the wrong saved-session-ID location; it was
corrected to the existing request.session_id contract before final validation.
No application/evaluator behavior, model input policy, old plan or trial result
changed. No inference, account change, Duodoro edit or release-gate closure.

CLI-lifecycle validation: 89 focused and 1,495 offline tests passed (five live
excluded), Ruff/format/mypy passed and 40 local documentation links resolve.
The existing Chroma warning remains. Application/evaluator behavior is unchanged,
so the preceding lazy-capture PR's two 90/90 scripted matrices were not rerun here.
