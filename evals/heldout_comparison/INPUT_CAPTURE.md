# Offline production-adapter first-input capture

`uv run python -m evals.heldout_inputs --directory /tmp/new-unique-input-directory`
walks all sixteen fixed cells through the shared runtime and production Ollama
adapter. A mandatory `httpx.MockTransport` intercepts `/api/chat`, records its
actual body and raises before forwarding. No socket/backend/model inference is
used. The deliberate `CaptureComplete` stop is preparation, not a model attempt,
provider failure observation or acceptance outcome. The original grader may run
on the unchanged disposable fixture; those results are not model evidence.

The exclusive mode-0700 directory contains sixteen normalized JSON requests and
sixteen exact wire bodies, all mode 0600. A final index binds their SHA-256 hashes,
plan/seal identity, observed runtime Git identity and safe runtime configuration.
Only the final index indicates capture completion. Failures preserve partial
artifacts without an index; reuse is refused. Reads/writes are bounded, not atomic
snapshots or crash-safe storage. Existing Git dirty status stays explicit.

Raw repeated inputs currently differ: the shared runtime includes the fresh random
disposable workspace root in the system prompt. `repeat_bytes_equal` records raw
comparison outcomes; capture does not mask or normalize those paths. An initial
capture probe refused this difference and remains retained separately. This is a
freeze obstacle: a later execution recipe must pin actual per-cell workspaces and
first-input bytes or explicitly review another path policy. Do not treat these
mock captures as reusable exact first-input approval for random future copies.

## Actual execution freeze remains incomplete

This bundle supplies a missing prerequisite, not the promised launch freeze.
Still required: reviewed workspace/input-path policy and execution recipe, actual
clean retained runtime and full resolved settings, current owned model-image and
backend version/endpoint pins, no-forward refusal probes, resource lifecycle and
settlement checks, independent exact-freeze review/CI/merge, parent approval receipt
and fresh launch gates. The generic CLI and scripted rehearsal are not approved
live launchers. No live inference, release approval or model-quality claim follows.
