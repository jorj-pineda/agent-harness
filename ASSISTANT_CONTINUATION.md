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

All current adapters reject supplied continuation state before HTTP rather than
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
