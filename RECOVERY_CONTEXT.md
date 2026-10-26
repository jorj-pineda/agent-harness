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
