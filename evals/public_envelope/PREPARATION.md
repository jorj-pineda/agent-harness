# Fresh public-pilot offline preparation

`prepare_public_candidate` is an internal one-shot API for a clean retained source
revision, started with the selected existing absolute external Python interpreter.
It reuses the existing interpreter guard: the retained tree may not contain a
local `.venv`, interpreter aliases do not substitute for the selected environment,
and the interpreter/prefix must stay outside that retained tree. The API never
runs a resolver, installer, service, CLI launcher or inference request.

The caller explicitly selects executable paths (including that Python and the
existing Ollama binary), dependency files, an existing source model directory and
the replacement command environment. Runtime/dependency observations and source
selection are checked before exclusive output allocation; selected inputs may not
overlap the destination. This observes only selected files and package versions,
not a complete import, shared-library or host freeze.

Preparation records private runtime/dependency reports, creates a verified owned
image through the existing copy helper, and captures all six shared-runtime first
inputs through mandatory MockTransport. The capture is preserved in an archive;
fresh unadmitted fixtures are staged at the same paths. The distinct public
manifest binds plan, capture, Settings/environment, runtime/dependency/image/binary
pins and canonical paths. Read-only public preflight, interpreter and clean-source
checks precede candidate manifest and preparation-receipt writes.

The receipt explicitly records no inference, no issued parent approval and no
verified import closure. Call the API separately for independently named candidate
and rehearsal directories; their fixtures, first-input roots and captures remain
separate. An occupied directory refuses instead of being reset. Failure after
allocation retains partial evidence and attempts withdrawal plus failure records,
without replacing the primary exception if recording fails. A candidate manifest
retained during a later receipt failure is withdrawn, not ready for execution.

Tests use a synthetic image-copy/verifier seam with real capture, staging and
preflight files. They establish preparation and refusal mechanics. Actual native
image operations and owned rehearsal/settlement evidence remain separate gates.
No preparation receipt grants approval or inference authority. A fresh separate
full owned mock rehearsal, refusal probes, applicable local/CI gates, concrete
exact-head freeze review/merge, actual parent receipt and fresh launch/resource
gates remain before pilot generation. No old attempt is resumed or relabeled.
