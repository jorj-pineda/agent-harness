# Explicit recovery-context preview

`harness.recovery_context.build_recovery_context` joins a saved recovery review
with deliberately selected historical source spans. Retained Beam attempts stopped
at context/total-token gates even when their artifacts passed acceptance; assembling
reviewable smaller evidence is a prerequisite for a later recovery experiment.
This slice supplies deterministic assembly only. No provider history is replaced,
no system prompt is generated and no continuation is dispatched.

The selection pins the complete validated run-record serialization SHA-256 (from
`recovery-review.source_record_sha256`) and zero to four explicit checkpoint-span
selections. Each names a canonical relative path, whole-file hash and inclusive
line range under the existing [span limits](CHECKPOINT_SPANS.md). Same-file ranges
must be disjoint and use one hash. Input order is retained, including disjoint
ranges supplied out of order; nothing is automatically ranked, merged or selected.
Zero spans supports review of legacy or failed work without a final checkpoint.
It does not imply that source inspection is unnecessary.

Only a terminal run's linked `turn_after` checkpoint can supply source. Its session,
run ID and paired-before ID must match the saved run before any blob is read.
A manual, before-turn, foreign or unrelated checkpoint cannot be substituted.
The saved-record hash is checked before checkpoint access. Callers must enforce
run ownership before invoking this helper. Checkpoints are immutable in ordinary
application operations; this does not protect against an operator rewriting SQLite.

The output embeds the complete existing bounded recovery-review packet and exact
selected spans, retaining their hashes, original status, missing-final unknowns,
truncation, omission counts and historical-source scope. A combined evidence hash
covers compact review JSON, a newline, and compact span JSON joined by newlines,
in selection order. It identifies this representation, not current state or truth.
Total selected source is at most 16,384 UTF-8 bytes and the complete compact output
at most 65,536 JSON bytes, including escaping/metadata. Any failed range, encoding,
identity, corruption or aggregate bound refuses the whole bundle without returning
a partial selection or silently dropping evidence. Rendering whitespace can enlarge
separately displayed output. Original source and run records remain unchanged.

This is untrusted, potentially sensitive owner evidence. All unselected files and
lines remain outside the preview; a selected span does not certify that instructions
or a whole file were read. Existing diffs are bounded per-turn review evidence,
not a reconstructed current patch. The bundle does not inspect current files,
load instructions/facts, estimate a prospective provider request, validate selected
configuration, alter/reset budgets or grant permission to execute. It is explicitly
`read_only=true`, `execution_supported=false`. Context fit, preservation of required
instructions and guarded execution need a separate design and validation. No
measured quality/recovery gain, release approval, default change or closure of the
broader compaction milestone is claimed. API and CLI access follow separately.

Fifty focused assembly/recovery/span regressions and Ruff/format/mypy pass.
Coverage includes exact saved source after restart without the original workspace,
failed checks and incomplete state, missing-final unknowns, pre-blob identity guards,
overlapping/excess selections, aggregate source/escaped JSON limits and all-or-error
selection. The initial test fixture incorrectly used manual checkpoint storage for
turn checkpoints; it was corrected to exercise real atomic run binding. The full
offline gate is required before merge; no live model result is claimed.

## Owner-checked API preview

`POST /runs/{run_id}/recovery-context` computes a read-only preview. POST carries
explicit structured range selections; it creates no run, checkpoint, journal or
saved bundle. The body contains `user_id`, `expected_record_sha256` and `spans`
(an explicit list of zero to four selections). Extra fields, including execution
options, are rejected. For example:

```json
{
  "user_id": "dev1",
  "expected_record_sha256": "<recovery-review source_record_sha256>",
  "spans": [
    {
      "path": "src/example.py",
      "expected_sha256": "<saved whole-file sha256>",
      "start_line": 1,
      "end_line": 20
    }
  ]
}
```

Placeholders above must be replaced with actual 64-character lowercase SHA-256
values. First review the existing owner-scoped recovery packet and run metadata;
obtain file hashes from that run's `after_checkpoint_id` metadata. This route never
accepts an arbitrary checkpoint ID or auto-selects/expands ranges. No current
workspace needs to exist. A repeated preview recomputes the same historical result
without dispatching work, but clients should report errors without automatic retry.

Existing user-ID ownership lookup runs before assembly/source access; this is the
project's existing ownership convention, not account authentication. Missing runs
return 404, foreign owners 403, invalid/missing body fields 422. Running records,
stale run/file hashes or missing/mismatched/corrupt checkpoint evidence return 409.
Uncaptured paths return 404, nonexistent ranges 422, invalid selected UTF-8 415,
and source/span/aggregate/encoded-response limits 413. Context errors and source
refusals return fixed codes with no partial evidence; SQLite errors return a generic
503. Invalid stored records return a generic 409 without private exception text.
Successful source responses set `Cache-Control: no-store`.

HTTP/SSE scripted real-file runs return byte-identical previews after application
lifecycle restart despite later user edits and a moved workspace. Original run,
archive, checkpoint and run count remain unchanged; preview builds no tools or
provider views. These checks establish historical assembly mechanics, not process
restart reliability, current-file attestation or live-model recovery. CLI access,
prospective prompt/context-fit review and guarded use of smaller history remain
separate work. All existing no-go and task acceptance criteria stay unchanged.

## CLI selection and preview

Save a deliberate selection as a UTF-8 JSON file containing exactly
`expected_record_sha256` and `spans`, using the API example above without `user_id`.
Obtain the run serialization hash from `recovery-review` and file hashes/ranges
from that run's linked final checkpoint. Then:

```sh
agent-harness recovery-review <run-id> --user-id dev1 --json
agent-harness recovery-context <run-id> --user-id dev1 \
  --selection-file /path/to/selection.json
```

Add `--json` to print the server preview. Human output leads with original run,
completion/check/workspace status and all recovery omissions/unknowns, followed by
selected source, combined hash and explicit non-execution scope. Both modes escape
saved text for terminal safety, including source terminators and control characters.
Displayed indentation/escaping can exceed the compact server response size.

The command reads only the owner-supplied selection file: regular UTF-8 input
capped at 8,192 bytes plus an overflow sentinel, with POSIX final-component
no-follow/nonblocking flags where available. Duplicate JSON keys,
extra fields, invalid hashes/ranges, overlapping selections and excessive selection
counts reject before HTTP. The API remains authoritative for canonical/excluded
paths, ownership and saved checkpoint access. There is one 30-second preview POST,
with no implicit metadata GET, retry, fallback, current-source read or continuation
submission. Private input/transport/server failures print exception category only,
plus explicit review guidance; file contents and error bodies are not echoed.

Before printing, the thin client reuses existing recovery-review and span response
validators, checks the requested run hash and ordered selections, final-checkpoint
kind/common identity, same-file metadata consistency, aggregate source bytes and
the combined evidence hash. Unsupported/inconsistent envelopes fail without partial
output. These are consistency checks against the selected response, not independent
verification that storage is truthful or that the checkpoint belongs to the run;
server-side binding remains necessary. A self-consistent hash does not authenticate
text, prove current state, validate context fit or grant execution permission.

The 210 focused core/API/CLI regressions include 80 new CLI cases. Scripted CLI
requests against a restarted real API preserve exact historical source and original
run/archive identity while later user edits and a moved workspace survive; no new
run or provider call occurs. Existing review/span commands pass after shared client
validation extraction. One test's overbroad privacy assertion matched the generic
word `saved` in error guidance; it was narrowed to detect actual evidence leakage.
Full offline/static gates remain required before merge. No inference, quality gain,
automatic compaction, execution or release-gate closure follows. This closes only
the three-PR deterministic assembly/API/CLI slice. Prospective prompt assembly,
instruction retention, context-fit review, guarded execution and measured recovery
remain separate work; panel controls remain deferred.
