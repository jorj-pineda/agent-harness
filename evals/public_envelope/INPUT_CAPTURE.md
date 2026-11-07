# Offline public-pilot first inputs

`capture_public_inputs` prepares the six named public cells and uses the shared
run_task path with the production Ollama adapter. Its transport is mandatory
HTTPX MockTransport at capture.invalid: it intercepts the first POST /api/chat,
preserves normalized and native request bytes, then deliberately stops before
forwarding. It accepts no caller transport override. No backend, model load,
metadata request, approval receipt or real generation is performed.

The caller supplies an explicit replacement command environment, selected runtime
inventory/hash and selected dependency-file manifest/hash. Preconditions check
the reviewed plan, clean source revision, Settings serialization and those pins
before allocating an exclusive capture directory. Capture and the public task
guard share the same selected policy fields. Complete validated Settings and the
environment are saved privately; runtime/dependency reports are preserved and
reobserved around cells. Selected files and package versions do not establish
complete import/shared-library closure or later host identity.

Each cell consumes its prepared fixture only to reach the mock boundary. The
capture retains its prepared receipt, admission marker and unchanged workspace.
Normalized/wire inputs are saved at that boundary, before later drift checks can
fail. Only the expected CaptureComplete stop and one request in each representation
can advance capture. Adapter cleanup runs on failure too. Partial artifacts remain
and occupied directories refuse; no reset or replay is performed.

The distinct public-pilot-inputs-v1 index is written last after all six cells,
source/runtime/Settings/environment checks, listed-file hashes and private modes,
fixture digests and bounded container inventory checks pass. The index pins each
normalized/wire/runtime-config file and both cell receipts, plus the complete
Settings, environment, selected runtime/dependency reports and clean revision.
No existing held-out or residency index is relabeled or treated as equivalent.

These are offline captures, not coding attempts or acceptance results. Controls
run on disposable fixtures through the shared evaluator, but a deliberately failed
provider call does not assess model capability. The expected model/version/digest
are protocol selections, not observed backend identity. Later verification,
capture-preserving staging, full mock rehearsal, owned backend/resource checks,
concrete reviewed freeze, actual parent receipt and fresh gates remain required
before pilot generation. Observations are bounded and non-atomic, not isolation,
continuous monitoring or authenticated launch authority.
