# Fresh twelve-cell execution selection and admission

`FreshExecutionManifest` is a distinct frozen validated boundary binding the exact
fresh selection, twelve-cell mock capture, full Settings/environment/inventory/
dependency hashes, image receipt, executable bytes, paths and existing policy.
Its `inference_authorized` field stays false: selecting artifacts is not authority.
Old campaign manifests and implicit/default paths do not select this cohort.

`select_fresh_execution` reads existing independently selected artifacts, constructs
the manifest and runs read-only preflight. It returns a model in memory, writes no
output and admits no cell. Expected runtime commit, capture index, image receipt,
binary hash and output path are explicit caller selections, not inferred approval.
The caller must retain exact manifest bytes before obtaining any parent receipt.

Preflight requires canonical disjoint evaluation-owned paths outside Git, unused
output, twelve unadmitted exact-path fixtures and preserved archive correspondence.
It reobserves clean source revision, complete selected Settings and command
environment, selected runtime/dependency files, verified model image and executable
hash/type/mode. The running Python entrypoint must exactly match the selected path;
resolving two venv launchers to the same base binary is insufficient. These checks
are neither complete import/shared-library closure nor authenticated provenance,
process isolation, a continuous ownership lease or execution approval.

A separate `FreshParentApproval` structurally binds the manifest/runtime and exact
reviewed/merged freeze commits. True review/CI/merge/inference fields do not prove
those facts or authenticate the issuer. `load_fresh_execution` only selects private
bytes and checks schema/hash correspondence; it does not issue or verify authority.

`admit_fresh_execution` requires a trusted-parent verifier with no default. That
caller must independently establish actual user approval, exact freeze review/CI/
merge and launch scope. It receives deep copies so mutable nested dict changes
cannot alter effective execution inputs. Parent rejection, source drift, withdrawal
or preflight failure precedes exclusive output reservation. Selection files must
remain separate canonical files outside all managed trees.

Only after those checks does admission exclusively reserve output, preserve the
manifest and parent receipt, recheck their source/preserved hashes and withdrawal,
and write its success record last. Failures keep partial output and the primary
exception, without retry/resume/reset. Two competing admissions cannot reserve the
same output. Fixture markers are not consumed by execution admission; the next
runner must retain result-before-cleanup state and enforce each cell's single use.
`check_fresh_admission` rechecks source/preserved byte pins and withdrawal for later
execution guards, without constructing providers or backends.

Focused tests cover explicit read-only selection, schema/cross-campaign refusals,
independent expected hashes, source/host/environment/image/binary/path drift,
Python alias refusal, parent rejection before allocation, callback deep-copy
isolation, post-preflight source drift, partial preservation, competing admissions,
selection aliases, withdrawal before/during admission and ongoing byte drift. Real
fresh captures/staging use the shared evaluator; host image/source seams are
explicitly synthetic where patched. Affirmative fixture receipts are never actual
authority. No model request or real execution approval occurs in these tests.

Next: compose admitted source/first-input/native-response/owned-transport guards
and a fixed twelve-cell serial owned runner. A concrete retained freeze with full
candidate checks, actual approval and fresh launch gates still precedes generation.
This code changes no default, grading criterion, prior cohort, model-quality claim
or release gate. It is not an executable launch recipe.
