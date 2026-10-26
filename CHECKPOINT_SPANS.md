# Exact checkpoint spans for recovery review

A recovery review packet contains saved objectives, diffs, failed checks and recent
actions. Preparing a smaller context also needs deliberately selected file spans.
`harness.checkpoint_spans.read_checkpoint_span` supplies exact ranges from an
existing saved checkpoint without reading or changing the current workspace.
This shared extraction slice precedes owner-checked API and CLI access. It does
not build or execute a compacted prompt.

Selection requires a canonical relative POSIX path, expected full-file SHA-256,
and explicit one-based inclusive start/end lines. Path aliases, traversal,
absolute/backslash/control-character paths and the shared checkpoint ignore
patterns (including environment files and ignored directory components) are
rejected. Callers must enforce checkpoint ownership before invoking the helper.
A requested file must be in the captured manifest and match the expected hash.

The helper refuses empty files and source files larger than 1,048,576 bytes before
allocating a blob. Existing checkpoint storage validates full saved length/hash,
including bytes outside the range. The range must exist in full, span at most
200 lines, and contain at most 8,192 UTF-8 source bytes. Lines use the file-tool
LF-delimited convention; original CRLF and selected final terminators are retained.
Selected text must decode as strict UTF-8; invalid bytes outside the chosen range
are not returned or lossily decoded. No range is silently shortened or truncated.
An empty file has no selectable line and is explicitly unavailable.

The response identifies session, checkpoint/kind, selected path, saved file mode,
full file hash/byte length, total lines, selected endpoints, exact content and
content hash/byte length. `has_more` reports whether lines follow the selection;
it does not assert earlier lines were read. A hash of the validated checkpoint
record serialization identifies the metadata representation, not database bytes,
file correctness or current state. The complete compact response must fit 32,768
UTF-8 JSON bytes; escaping or oversized custom metadata can cause explicit refusal
even when source-byte limits pass. Rendered/indented output may be larger.

Refusals have fixed codes for uncaptured file, expected-hash mismatch, source/span/
response limits, unavailable or inconsistent content, nonexistent range and invalid
selected UTF-8. They return no partial text or exception/source payload. SQLite
I/O failures remain separate storage errors for transport callers to handle.
Manual, before-turn and after-turn checkpoints retain their original meanings;
selecting text does not make a manual snapshot a finalized run artifact.

Checkpoint source contents are untrusted owner evidence and may contain sensitive
source text. This is an explicit content review primitive, separate from existing
metadata routes, not a public redacted export. No provider, model, tool execution,
filesystem resolution, checkpoint rewrite, history repair, budget reset or
continuation/revert permission is involved. Later edits and missing/moved original
workspaces do not change the selected historical bytes. Full originals stay in
checkpoint storage; selected ranges are not a substitute for unread instructions
or authoritative checks. Actual compaction, current-file selection and guarded
execution of a smaller context remain separate work. Existing no-go, 8/10 targets
and model-quality limits are unchanged.

Validation: 37 focused regressions and 1,638 offline tests (five live deselected),
Ruff/format/mypy pass. Cases cover exact Unicode/CRLF/final terminators, immutable
review after restart without the original workspace, strict path/range inputs,
metadata refusal before blob reads, empty files, invalid selected UTF-8, source/
span/escaped-response limits and corruption outside the selected range. These
validate deterministic evidence access, not live-model recovery or compaction.
