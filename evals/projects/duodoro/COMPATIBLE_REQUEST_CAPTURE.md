# Bounded compatible-provider input recording

The evaluation-only recorder can explicitly select `/v1/chat/completions` or
`/inference/go/v1/chat/completions` alongside its unchanged `/api/chat` default.
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

Actual OpenAI SDK requests against MockTransport cover both explicit compatible
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
