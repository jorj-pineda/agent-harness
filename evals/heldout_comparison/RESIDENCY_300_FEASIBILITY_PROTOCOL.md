# One-cell 300-second residency feasibility

Jorge requested a fresh small test after the fixed residency comparison stopped
before its 300-second arm. This separate protocol selects exactly one fresh
ISO-week harness attempt with 300-second keep-alive. It changes no original task,
criterion, model, runtime prompt/tool setting or per-cell budget. It is development
on retired v1 material, not a paired comparison or held-out evaluation. This
document supplies no inference authority; the concrete reviewed execution freeze,
local gates, both CI checks, merge, actual parent receipt and fresh launch checks
are still required.

## Scope and preserved limits

Only original residency ordinal 2 is planned: ho_iso_week, harness, repeat 1,
keep-alive 300 seconds. Use a fresh evaluation-owned workspace and independently
verified image, empty loaded-model startup and its own owned backend lifecycle.
There is no warmup, preliminary inference, second attempt, control cell, manual
patch repair, substitution or retry. The earlier closed cohort remains unchanged.

Use the same local Ollama 0.33.2/gemma4:12b digest and original plan.json settings:
150 dispatch seconds, 32K total tokens, 4096 completion tokens, 12 tools, 6 iterations,
request cap 2048/min 256, context 16384, thinking off, temperature zero and the
original visible check and external acceptance. Setup, grading, preservation and
cleanup lie outside dispatch time; there is no hard end-to-end deadline. No
dependency/model downloads or installs, Docker, hosted/paid calls or user Ollama
service/configuration management. Inspect actual processes/listeners first;
only evaluation-owned 11447 may be managed, preserving user 11434 and identity.

## Explicit single-cell admission

Reuse the existing strict sixteen-input offline residency capture/staging bundle
to preserve tested first-input, Settings, environment, runtime, dependency,
catalog/criterion and fixture guards. Mock capture consumes fresh preparation
copies only; staging archives those and prepares fresh exact-path fixtures.
The other fifteen captured fixtures are verification inputs, not planned live
attempts or retry slots. They must remain unadmitted and unchanged.

The manifest must explicitly bind residency_feasibility_ordinal=2 along with the
original residency schedule hash and no global residency override. This changes
its bytes and cannot reuse a previous approval hash. Preflight revalidates the
schema, refusing constructed invalid selections before path/resource inspection.
The full cohort entry point refuses a feasibility manifest; the feasibility
entry point refuses a full-cohort manifest. The provider factory refuses any
ordinal outside the selected cell before inspecting pins or ownership.

The shared executor uses a distinct residency-feasibility-ledger-v1 with exactly
one planned cell, original ordinal 2. It retains the existing exclusive start
marker, shared runtime/tools/grading, independent image preparation, native
response preservation, request guards, post-response owned GET observations and
cleanup. Results persist before settlement; only a clean settled cell becomes
recorded. A stopped started cell can retain its result. No loop or provider is
duplicated. Other captured cells never become live work through this selection.

## Stops, evidence and interpretation

Unknown usage/submission, provider/transport/model/runtime/environment/criterion/
preservation failure or failed/forced cleanup closes the attempt. Cancelled
generation retains its real cause and unknown usage, without being mislabeled
as observation failure. Real observation errors/cancellation retain that cause.
There is no continuation, reset or resubmission after any terminal outcome.

Report whether the model is observed loaded after complete responses, whether
it reaches editing, original final-artifact acceptance, native completed-and-
accepted status, tool refusals, checks, usage known/unknown and manual intervention.
Preserve complete-response timing prefixes and cancelled tails as unknown. A
loaded snapshot is not continuous-residency, cache or GPU-settlement proof. A
single result cannot establish a causal residency effect or a supported coding
envelope. No default, broader quality claim, held-out gain or no-go/8-of-10/G2–G5
closure follows. Publish the result and retained evidence even if unchanged or
rejected; any later experiment requires a new decision.
