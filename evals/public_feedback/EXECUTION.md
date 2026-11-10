# Explicit feedback execution selection and admission

This slice binds the eight-cell comparison to selected execution inputs. It provides
no executor, live CLI, provider factory or approval generator. Existing held-out
and six-cell public-pilot paths retain their distinct formats and behavior.

## Selected manifest and structural receipt

`FeedbackExecutionManifest` uses `public-feedback-execution-v1`, binds the exact
feedback plan and selected capture/Settings/environment/runtime/dependency/image/
binary hashes, and requires explicit absolute paths. Direct/whole_file_check and
keep-alive 300 remain fixed. The effective replacement command environment must
include its pytest startup flags; credentials and invalid environment values refuse.
The manifest retains `inference_authorized: false`.

`FeedbackParentApproval` uses `public-feedback-parent-approval-v1`. It binds the
manifest/runtime and exact reviewed/merged freeze commits, with affirmative review,
CI, merge and inference assertions. `load_feedback_execution` validates selected
private bytes, schemas and matching manifest/runtime references without allocating
resources. **These assertions and hashes do not authenticate an issuer or prove
review, CI, merge, user authority or current resource ownership.**

## Read-only preflight

`preflight_feedback` revalidates the schema, canonical/nonoverlapping owned paths
and absent output. It verifies all eight fresh unadmitted staged cells and the
preserved first inputs, compares their runtime/artifact pins, reconstructs complete
shared Settings independently, and checks the effective command environment.

Unlike archival checking, this gate reobserves selected executable/dependency
inventory inputs and the clean source revision. It verifies the prepared model
image through existing bounded image checks and the selected executable's type,
size, executable mode and content hash. Source/path/fresh-cell checks are repeated
before return. It creates no output, consumes no cell and starts no service.

This is a preparation observation, not approval, a resource lease or a hard
end-to-end deadline. Selected dependency files do not establish complete import or
shared-library closure. Backend version, actual loaded model metadata, process
ownership, native delegation guards and clean settlement still belong to the
future owned execution/rehearsal path.

## Single-use cohort admission

`admit_feedback_execution` requires explicit manifest/receipt paths and hashes and
a **mandatory trusted `verify_parent` callback with no default**. It checks selected
source locations are canonical, separate and outside managed trees. The callback
receives deep copies and must refuse unless actual authority and the exact reviewed/
merged freeze/CI prerequisites have independently been established. It is trusted
operator code, not a model judge or JSON authenticator.

Parent verification precedes preflight and output allocation. After those gates,
an exclusive directory reservation consumes the selected output. Exact manifest
and receipt bytes are preserved privately, checked again, and a success record is
written last. Failure retains partial evidence without rollback or retry; a
secondary recording failure does not replace the primary exception. Competing
admissions reserve at most one output. Any withdrawal marker refuses without
opening its contents, including markers appearing during admission.

`check_feedback_admission` rechecks selected and preserved manifest/receipt bytes
and withdrawal. It is an ongoing provenance primitive, not a complete input/resource
or already-running-cell guard. Admission reserves the cohort output only: no cell
is admitted and no inference occurs. Paths and private modes are not OS isolation.

## Proof

Fifty-seven focused regressions pass, covering scope/schema/receipt refusal,
preflight source/input/path/image/binary drift, parent refusal ordering, withdrawal,
deep-copy isolation, partial preservation, competing admissions and continued
selected/preserved-byte checks. Synthetic preflight tests stub source revision and
model-image verification while using real captures, staged files and other checks.
Positive parent callbacks are explicitly scripted fixtures, not real approvals.

A separate **native read-only preflight** passed from clean source
`ef9e6f9a0a7ae45a1ab5d13f128644cc1fb765f2`. A fresh mandatory-mock capture was
staged; a new APFS copy of the existing selected model image and actual existing
Ollama executable bytes were verified without starting them. All eight cells
remained fresh/unadmitted and no execution output was allocated. Real HTTP
transport construction was forbidden during the proof preparation. No parent
receipt was issued or admission performed by this native proof.

The [summary](preflight-proof-20261110.json) records exact selected hashes and
limitations. Private recipe, manifest, inventories, copied image and preserved
capture remain at:

`/Users/jorge/.local/share/agent-harness/trials/public-feedback-mock-_5l62piu`

That proof is historical to its named clean revision. A later source head needs
fresh matching preparation; these files are not a reviewed execution freeze or
approval for a live run. It establishes neither backend version/ownership nor
complete dependency closure or model quality. No downloads, installs, user-service
changes, model inference or paid calls occurred.

Next: explicit admitted provider guards and owned serial execution, followed by
fresh retained preparation/full owned rehearsal, concrete freeze review/CI/merge,
actual approval and fresh launch gates. No old cohort, default, original acceptance,
release target/no-go or G2–G5 status changes.
