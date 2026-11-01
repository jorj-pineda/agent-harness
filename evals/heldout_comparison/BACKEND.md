# Owned metadata-only backend rehearsal

This is a preparation check, not the comparison launcher:

```sh
uv run python -m evals.heldout_backend_rehearsal \
  --image-directory /absolute/path/to/new-image-container \
  --directory /absolute/path/outside/git/new-backend-rehearsal \
  --binary /absolute/path/to/installed/ollama
```

Use a fresh image from `heldout_model_copy`. The container receipt, exact model
inventory, read-only file modes and all pinned bytes are checked before spawning.
Unknown or changed entries refuse. The new backend directory must be absent and
outside Git workspaces and the image. The installed executable is resolved and
hashed; no download or installation is attempted. Ports and paths are explicit:
only literal loopback port 11447 is managed, never the user's default 11434.
An occupied port refuses before allocation without an HTTP request or signal.

The child receives an explicit small environment, including its own mode-0700
HOME and temporary directory, the prepared model path, cloud/pruning/history off,
keep-alive zero and one model/parallel request/queue slot. Ambient environment
variables and credentials are not inherited. Ollama may create its own identity
inside this new HOME; no user identity/configuration file is read or copied.
Environment, process identity, logs and receipts are retained privately. The
server log is a private file, not a bounded model-visible trace.

The foreground child starts in a separate process group. Before metadata GETs,
`lsof` must identify exactly that live child as the listener. Startup allows thirty seconds for listener and metadata readiness. A bound
listener can precede GPU discovery; only metadata ReadTimeout/ConnectError may
retry, with each error retained separately and at most sixteen retries. These are GETs, never generation
retries. Exit, invalid/mismatched metadata, unknown ownership or foreign PID refuses.
The deadline is checked between probes; individual metadata timeouts/cleanup
can extend total duration. The only HTTP operations are GET /api/version, /api/tags and /api/ps,
using bounded bodies, timeouts, no proxies and no redirects. Protocol version and
selected model digest must match, and loaded models must be empty. Ownership is
checked again after observation. There is no generation, pull, warmup, unload or
live-provider path in this utility.

Cleanup runs after every post-spawn failure or interruption, including recording
and registration failure. Only a freshly verified original process group is signaled, including before
a force kill. Poll, group-observation, signaling and wait errors are retained;
they cannot skip an independent read-only settlement-gate attempt. Unknown group
identity refuses signals. Cleanup evidence separates observed settlement from a
passing cleanup; even a settled child cannot erase an earlier cleanup error. Graceful group termination gets twenty seconds; a necessary force
kill gets five more and cannot yield a passing rehearsal. The shared ownership
gate requires exit zero, no surviving group members and an available port.
Unknown or failed settlement is recorded as non-passing and must stop preparation;
no foreign listener is stopped to free the port. Failure artifacts and partial
state remain without reset/retry. A result receipt is written only after cleanup
and final image/executable rechecks; receipt absence is incomplete preparation.

File checks, port availability and PID observations are bounded non-atomic host
observations, not isolation, port reservations, leases or exclusion of external
writers. The local service exposes its normal API while alive; trusted local
processes can interfere. This metadata rehearsal grants no generation capability
or launch authority. It does not demonstrate model inference, cancellation of a
real model request, backend compliance under inference, or later freshness. Do
not substitute it for the full operator/refusal/settlement recipe, exact reviewed
execution freeze, parent approval receipt and fresh launch gates. Never reuse the
closed historical comparison or its approval. A future live launcher must own
its processes and enforce those separate gates.

Offline tests use a tiny HTTP child and synthetic model files, with listener-PID
observation substituted so CI needs no lsof installation. They exercise actual
process groups, signals, port settlement and the metadata-only request sequence;
they do not prove Ollama or APFS behavior. Native preparation and metadata-only
Ollama rehearsal remain separate recorded environment observations.

## Shared synchronous ownership context

owned_backend(image_directory, fresh_directory, installed_binary) now owns the
same validated startup and cleanup path used by the metadata-only CLI. It yields
OwnedBackendSession only after the selected image, executable, listener ownership
and empty-model protocol metadata checks pass. check_owned() can serve as the
raise-on-refusal callback for OwnedInferenceTransport. It refuses outside the
active context and re-observes child/group/listener ownership each time.

Enter and exit outside an active event loop; misuse refuses before image access
or process creation. A future synchronous cohort owner can enter the context,
run its separately approved asynchronous cell, and then exit for settlement.
The context is deactivated before cleanup. Caller exceptions are recorded and
still run owned cleanup; no successful session receipt is written on that path.
Normal exit rechecks image/executable bytes and records session.json only after
settlement. The session record makes no generation-count claim because resource
ownership alone cannot observe arbitrary caller work. The metadata-only wrapper
retains its own result format and zero-generation count for its empty body.

This context grants no inference permission or reusable ownership lease. A caller
must enforce the exact freeze/parent approval/fresh gates before submitting any
request, and must stop its cohort on failed or unknown settlement. Resource code,
callbacks and host processes remain trusted; there is no isolation or exclusion
of external writers. The live operator and approval/freeze gates remain unfinished.
