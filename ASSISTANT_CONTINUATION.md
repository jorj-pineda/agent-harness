# Bounded assistant continuation state

This is protocol preparation, not evidence of Beam compatibility or coding quality.
No inference ran. The current installed Go catalog advertises Beam's interleaved
`reasoning_content` field, tool calling and a compatible endpoint; metadata does
not establish the field's actual presence, requirement or backend behavior.

An offline real-SDK response with a synthetic reasoning marker and tool call
reproduces the existing boundary: the raw response retains the marker, but the
normalized assistant content/tool history omits it. Do not infer that Beam has
already failed for this reason. Do not reconstruct replay data from an answer or
inject it into instructions to compensate.

`providers.base.ContinuationState` now represents one explicitly named format,
`openai-reasoning-content-v1`, and exact text. It rejects unknown formats/extra
fields, non-text values and text over 65,536 UTF-8 bytes without coercion or
truncation. Empty text remains distinct from absent state. Only assistant
ChatMessage records may carry it. It is provider replay data, not answer text,
source evidence, an instruction or a guarantee of semantic correctness.

Absent state is omitted during message serialization, preserving old normalized
message shapes and request identities. Present state survives JSON round trips,
changes the normalized request identity and contributes serialized UTF-8 bytes to
the existing heuristic context/total-token estimate before dispatch. The size cap
is per record, not a database quota, exact tokenizer, backend budget or hard RSS
bound. State is plaintext and potentially sensitive, like existing transcripts.

At the boundary introduction (#216), all adapters reject supplied continuation
state before HTTP rather than
silently dropping it or sending it through another protocol. This slice adds no
response producer or replay option: ordinary messages and default response handling
remain unchanged. Explicit compatible normalization/round trips and durable shared-
loop coverage follow separately. No default model, prompt, retry, pricing, task
criterion, old plan or historical result changed. No hidden premium dependency,
Duodoro working-checkout edit, release approval or G2–G5 closure is introduced.

Focused offline tests cover exact empty/Unicode/boundary round trips, invalid roles/
formats/types/extra fields, legacy serialization, request-identity change, a context
stop before dispatch and no-forward rejection across all three adapters including
a bound compatible-style view. A synthetic real-SDK fixture reproduces raw-only
reasoning retention without contacting a model.

Validation: 100 focused offline tests passed (four live deselected in that subset),
1,515 offline tests passed (five live deselected), Ruff/format/mypy passed. Both
scripted matrices passed 90/90 at `/tmp/agent-harness-continuation-boundary.S2mYj1`;
scripted-contract labels and metric limits were inspected. An import-order lint
finding was corrected before the final static gate. The first focused invocation
omitted the live filter; all four live cases skipped for absent configuration and
no inference ran; the final focused invocation explicitly excluded them.

## Explicit compatible replay

`OPENAI_COMPATIBLE_REPLAY_REASONING_CONTENT=true` opts the shared configured
compatible factory into this single protocol; it defaults false and is independent
of the named profile. Dedicated Ollama/Anthropic paths remain unsupported. Direct
SDK adapter construction may explicitly select `replay_reasoning_content=True`.
Go still uses its stable conversation headers and zero SDK retries.

On opt-in, absent/null native reasoning stays absent and an empty string stays
exactly empty. Text is validated through the same 65,536-byte boundary before a
normalized response can reach tool dispatch. Invalid types/overflow raise a generic
provider error without echoing the raw field. Usable assistant responses retain
state separately from content in shared history and optional normalized traces;
subsequent compatible assistant messages send the original text as
`reasoning_content`. Other messages receive no fabricated field. Default response
handling and old no-state serialization stay unchanged.

Replay state is counted by the existing pre-request heuristic. Reported output
usage remains authoritative; replay does not bypass missing-usage, truncation or
other completion gates. A large valid state can permit an already requested tool
and then stop the next request for context pressure; this is not a preventive
whole-response memory limit. No state is promoted into the final answer or source
citations. Provider continuation is not an instruction from the harness.

The allowlisted provider-options hash records enabled replay and its protocol/byte
policy without credentials or state text. Disabled legacy generic/Go recipe hashes
stay unchanged; enabling/disabling replay causes configuration drift for guarded
continuation. Other provider identities are unaffected. Real-task/simulated-tool
reports record the configured flag explicitly; historical reports stay unchanged.

Synthetic SDK/loop tests cover exact Unicode/empty/missing round trips, tool IDs,
bound session routing, JSON archive/trace round trips, malformed/oversized-state
no-tool rejection, state-driven next-request context prevention, unchanged output/
usage guards, default rejection and configured factory/profile drift. These are
mechanics, not actual hosted behavior or model-quality evidence. Durable API
restart and guarded-fork regression coverage follows in the third slice. Beam
selection/price/account checks and a new reviewed actual execution freeze remain
required before any replacement comparison; the old plan remains historical.

Compatible-replay validation: 144 focused tests passed (one live deselected in the
subset), 1,529 offline tests passed (five live deselected), Ruff/format/mypy passed.
Both scripted matrices passed 90/90 at `/tmp/agent-harness-compatible-replay.L1A6Vq`,
with contract labels/metric limits inspected. A routine test literal-style lint
finding was fixed before the passing gate; the existing Chroma warning remains.
No inference or model-quality comparison ran.

## Durable API lifecycle and guarded forks

HTTP and native SSE synthetic-SDK regressions now perform a real read, write and
configured pytest check, save all four exact assistant continuation records in
SQLite, and inspect identical source session/run JSON after a fresh application
lifecycle. A matching guarded fork sends that saved state under its own session
header without altering the source archive. Disabling replay changes the captured
configuration; changing workspace bytes changes the scoped baseline. Either blocks
the fork before a provider request or new checkpoint. Keyed fork replay does not
execute again. Ordinary restored sessions remain read-only.

Non-text/oversized first response state rejects the proposed write through HTTP
and SSE before mutation. The saved run stays failed with absent final response,
unknown failed usage, before-only checkpoint and initial-only session archive;
restart GET does not invent finished history. Pre-existing/later bytes and modes
survive, owner separation holds and both SDK clients close.

These tests exercise the actual default Components/settings/shared runtime and
real SQLite/file/check operations, with only SDK HTTP intercepted. Named-profile
selection keeps other providers unconstructed. They use in-process TestClient
application restarts, not an OS-process/browser restart or hosted Beam. Passing
configured pytest is observed check evidence, not external Duodoro acceptance.
Failed/crashed turns still have the existing partial-transcript limits; this does
not repair missing history or prove safe arbitrary model switching.

The initial fixture omitted named-profile selection and tried to construct an
unused Ollama adapter with the SDK interception arguments; selecting the intended
profile fixed the test setup before final validation. An import-order lint finding
was also corrected. No product change was needed in this test/docs slice.

This completes the bounded state/replay/durable-coverage three-PR batch. Actual
Beam access/region/current price/account allowance, reasoning/tool behavior, a new
replacement comparison protocol, actual client preflight and an independently
reviewed execution freeze remain required. No inference, paid call, Duodoro edit,
old-result/criterion change, quality gain or G2–G5/release closure is claimed.

Durable-coverage validation: 68 focused and 1,539 offline tests passed (five live
deselected in the full gate), Ruff/format/mypy passed and 41 local documentation
links resolve. The existing Chroma deprecation warning remains. This tests/docs
slice changes no application/evaluator behavior; the preceding replay PR passed
both scripted matrices 90/90, and they were not rerun here.
