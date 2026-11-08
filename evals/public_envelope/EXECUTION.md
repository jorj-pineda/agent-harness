# Public-pilot execution selection and read-only preflight

The public pilot uses its own `public-pilot-execution-v1` manifest and
`public-pilot-parent-approval-v1` receipt schema. Old held-out/residency manifests
and receipts cannot substitute for these formats. The manifest selects the exact
reviewed public plan, clean runtime revision, capture index, complete Settings and
command environment pins, selected runtime/dependency reports, prepared image
receipt, executable hash, and explicit managed paths. The selected policy remains
`direct` / `whole_file_check`, six fixed cells and 300-second keep-alive; the plan
retains all original per-cell budgets and criteria. No single-cell or baseline
scope is added. The manifest itself always says `inference_authorized: false`.

`load_public_execution` reads bounded private manifest/receipt bytes by separate
caller-selected hashes, validates their strict schemas, requires the receipt to
bind that exact manifest hash and runtime, then rechecks both files. This is
read-only binding, not admission. Affirmative receipt fields are parent assertions,
not independently authenticated review, CI, merge or user authority. A trusted
parent must verify those facts and the actual freeze before issuing a real receipt.
The loader does not supply that verifier, issue receipts or create an executor.

`preflight_public` can check a candidate before approval. It revalidates the
manifest (including objects constructed without validation), refuses noncanonical,
symlinked, overlapping or occupied output paths before inspecting artifacts, and
checks the current clean runtime and all six fresh staged fixtures. The capture's
runtime and Settings/environment/report pins must agree with the manifest.
Capture and preflight construct the same init-only validated Settings through
`public_settings`, so host environment variables and `.env` do not override it.

Unlike archival verification, preflight freshly observes selected executable,
package-version and dependency-file evidence. It validates the prepared model
image through the existing image verifier and hashes the selected executable,
requiring a bounded regular executable file. Receipt, runtime, paths and complete
staging checks are repeated before returning the capture index. No output,
admission marker, child process, provider or backend is constructed.

The checks are bounded non-atomic observations, not isolation or a continuing
lease. Selected dependency files do not establish complete import/shared-library
closure; recorded metadata is not authenticated model identity. Synthetic tests
use a stub image verifier with real capture/staging/file checks, not native image
allocation or a live model. They establish refusal and binding mechanics only.

Single-use trusted admission, ongoing request-boundary guards, owned per-cell
execution/settlement, full mock rehearsal, concrete reviewed freeze and actual
parent approval plus fresh launch gates remain required. These APIs do not grant
inference authority or establish model-quality, causal or release improvement.

## Single-use trusted admission

`admit_public_execution` requires an explicit trusted `verify_parent` callback.
The callback must refuse unless actual parent authority and exact freeze review,
CI and merge have been independently established. It receives deep copies so
mutating nested values cannot change effective preflight settings. Structural
JSON assertions alone never establish those facts.

Admission checks withdrawal and selected manifest/receipt bindings, keeps their
canonical private source files outside managed trees, invokes the parent verifier,
runs public preflight, then exclusively reserves output. It preserves the exact
selected bytes, checks source and preserved pins again, checks withdrawal and
writes `admission.json` last. Competing or repeated admission cannot reuse output.
Post-allocation failures retain partial files and attempt failure evidence; a
secondary recording error does not replace the primary exception. No public cell
is consumed, and no provider or backend is constructed by admission.

`check_public_admission` rechecks source manifest withdrawal, source manifest and
receipt hashes, and both preserved copies. A later executor must call it at its
resource and request boundaries, and still enforce its own single-use start,
Settings/input/runtime/ownership checks and settlement. These calls observe bytes;
they are not an atomic revocation mechanism or authenticated receipt verification.
A public runner, full owned rehearsal, concrete freeze and actual parent receipt
plus fresh launch gates remain open.

## Admitted provider guard composition

`build_public_cell_provider` requires a public admission, an ordinal, the actual
Settings object, an explicit transport and an ownership-check callback. It checks
source/receipt preservation and withdrawal, compares the effective manifest with
its preserved bytes, checks current runtime/Settings/dependency pins and ownership,
and loads the exact selected public first-input pins. It makes no provider request
during construction and never supplies a default network transport.

The composition reuses SettingsPinnedProvider, PinnedOllamaProvider,
OwnedInferenceTransport and CheckedProvider. Normalized and native first requests
must match the selected bytes. The owned transport rechecks the guard before
request delegation and before returning the response; withdrawal or ownership
loss at either boundary refuses and latches closed. Existing native completion,
model, usage and received-HTTP refusal behavior remains shared. Model strings are
not authenticated model identity, and these boundaries do not establish continuous
ownership or isolation.

On construction failure, the caller owns closing the supplied transport. After
successful construction, closing the returned provider closes it. The factory
creates no backend, task admission, retry, result or settlement evidence. The
trusted caller still owns actual approval provenance and the full per-cell resource
lifecycle. A serial public runner, full owned rehearsal and concrete freeze/receipt/
fresh gates remain open. Mocked guard tests establish mechanics, not model quality.
