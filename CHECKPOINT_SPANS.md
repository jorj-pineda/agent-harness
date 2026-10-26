# Exact checkpoint spans for recovery review

A recovery review packet contains saved objectives, diffs, failed checks and recent
actions. Preparing a smaller context also needs deliberately selected file spans.
`harness.checkpoint_spans.read_checkpoint_span` supplies exact ranges from an
existing saved checkpoint without reading or changing the current workspace.
Owner-checked API and CLI access are available. This review workflow does
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

## Owner-checked content API

`GET /sessions/{session_id}/checkpoints/{checkpoint_id}/span` requires query fields
`user_id`, `path`, `expected_sha256`, `start_line` and `end_line`. Obtain the saved
whole-file hash from the existing checkpoint metadata route, then deliberately
select the path and inclusive range. This endpoint returns source text; the existing
checkpoint list/detail routes remain metadata-only. Successful content responses
carry `Cache-Control: no-store`.

The route checks the existing user-ID ownership convention and checkpoint/session
binding before reading a blob. This convention is not a new authentication system.
Unknown/cross-session checkpoints or uncaptured files return 404; other owners
return 403. Invalid selections/nonexistent ranges return 422; expected-hash mismatch
or inconsistent saved content returns 409; source/span/response limits return 413;
invalid selected UTF-8 returns 415. SQLite read failures return generic 503 errors.
Fixed refusal codes include no partial source or underlying exception text.

Manual and turn-bound checkpoints are reviewable with their original kinds. API
lifecycle restart, later user edits and a moved/missing original workspace do not
change historical spans. The GET never resolves current files, constructs tools,
selects a provider, captures new data, changes archives or starts a run. It supplies
historical evidence, not a current-workspace lease, automatic context selection,
compacted prompt or continuation/revert permission. Explicit source disclosure
retains the untrusted/sensitive owner-data limits above.

API validation: 50 focused regressions and 1,657 offline tests (five live excluded),
Ruff/format/mypy and local documentation links pass. Nineteen new endpoint cases
cover manual and real-file scripted turn-after checkpoints, exact restart responses,
later edits/missing original workspace, owner/cross-session and pre-blob selection
guards, fixed refusal codes, wire bounds and private SQLite lookup/blob errors.
These are scripted lifecycle checks, not live-model or OS-process recovery evidence.

## CLI selection

```sh
agent-harness checkpoint <session-id> <checkpoint-id> --user-id dev1 --json
agent-harness checkpoint-span <session-id> <checkpoint-id> --user-id dev1 \
  --path src/example.py --expected-sha256 <saved-file-sha256> \
  --start-line 1 --end-line 1
```

Obtain the full saved file hash from checkpoint metadata, then select a range that
exists in full. For an unfamiliar nonempty text file, an explicit first-line query
can reveal total lines before a separate deliberate larger selection. No automatic
metadata fetch, range expansion, retry or fallback occurs. Add `--json` for the
bounded source response; default human output labels historical source, checkpoint
kind, exact range, total lines, unread following lines, saved mode, byte counts and
hashes. Both modes JSON-escape source and metadata for terminal safety. Indentation/
escaping can enlarge displayed output beyond the compact response encoding.

The thin client checks supported v1 identity, requested path/hash/range, byte/line
counts, saved mode and following-line consistency before either output mode. It
checks the selected-content hash and exact LF line count. A claimed complete file
must match its full byte length/hash; a nonfinal range must retain its terminating
LF. Partial-range checks do not independently verify the unreturned whole file or
saved metadata. No harness/runtime logic or filesystem access is duplicated in the
client. Invalid hash/range inputs reject before HTTP; the API remains authoritative
for canonical/excluded paths and source access.

Private transport/server errors produce an exception category plus a reminder to
review metadata/hash/range, with no raw body or exception text. Unsupported or
contradictory source responses fail before printing content. Empty files, oversized spans or selected lines that are not UTF-8 remain
unsupported; narrowing the line range cannot fix an
oversized whole-file limit or a single oversized line. Select another supported
file/checkpoint rather than changing historical evidence. No capture, restore,
compaction, provider call or continuation is submitted by this command.

CLI validation: 109 focused regressions and 1,715 offline tests (five live excluded),
Ruff/format/mypy, installed command help and local documentation links pass.
Fifty-eight new CLI cases cover identity/count/hash/range contradictions, exact
content and escaping, private errors/refusals, pre-HTTP input rejection, wire limits
and real CLI GETs against a restarted application lifecycle. Historical text,
archive/checkpoint identity and later user edits survive with no new inference.
These checks establish review mechanics, not successful compacted-context recovery.
