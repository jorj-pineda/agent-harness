# Existing toolset duration observations — 2026-10-20 UTC

**Completed-and-independently-accepted: 0/2 full,
2/2 whole-file.** Artifact acceptance was
0/2 versus 2/2. Completion,
observed checks and external acceptance remain separate. These four observations
on one selected public task do not establish a general/causal improvement or a
reason to switch the production default.

[Structured observations](../../results/2026-10-20-duodoro-toolset-comparison.json)
retain every fixed attempt, numeric request-budget journals, edit-tool evidence,
unknowns and final audit hashes. The reviewed [protocol](TOOLSET_COMPARISON.md) and
[manifest](toolset-freeze.json) are unchanged. Reporting commits were not inference
runtimes. Original ten-task, feedback and batching outcomes retain their original
scope; G2–G5 remain unchanged/unmet.

## Every scheduled outcome

Tokens are reported prompt / output. Requests/responses are observed numeric
journal entries; `write_file` means a successful recorded write on the allowed
file, not independent patch acceptance. Seconds are saved run-created-to-finished
intervals, not pure inference or hard end-to-end deadlines.

| Attempt | External artifact | Completion / observed check | Requests / responses | Target write_file | Tokens | Run seconds |
|---|---|---|---:|---|---:|---:|
| 01-T01-r1-full | Fail | budget_exhausted / not_run | 7 / 7 | no | 51,323 / 832 | 167.16 |
| 02-T01-r1-whole_file | Pass | completed / passed | 8 / 8 | yes | 58,967 / 1,132 | 190.77 |
| 03-T01-r2-whole_file | Pass | completed / passed | 8 / 8 | yes | 58,967 / 1,105 | 185.42 |
| 04-T01-r2-full | Fail | budget_exhausted / not_run | 7 / 7 | no | 51,323 / 832 | 161.67 |

Saved completion reasons:

- `01-T01-r1-full`: Estimated next request of ~12196 prompt tokens does not fit the remaining total token budget (64000 per turn).
- `02-T01-r1-whole_file`: Completed; no terminal error reason.
- `03-T01-r2-whole_file`: Completed; no terminal error reason.
- `04-T01-r2-full`: Estimated next request of ~12196 prompt tokens does not fit the remaining total token budget (64000 per turn).

All initial tasks used the actual CLI, in full/whole-file/whole-file/full order,
with fresh sessions and independent unseeded stores. The entire common workspace
was preserved after each settled run; no artifact seeded a later baseline. No
operator coding hint, manual patch repair, replacement attempt or extra allowance.
All four CLI responses timed out at 120 seconds while server-owned work continued.
Original-key read-only saved-run inspection retrieved outcomes without another
coding submission.

## What differed and what stayed fixed

Both presets used published runtime `ce5d54c7394c476afcee29a8981fecd96631cc9f`,
compact prompts and the same literal root. Whole-file mode removes only
`replace_text` and selects its existing full-content `write_file` instruction.
Retained schemas, other settings, sampling and complete instruction/tool-result
retention match. Initial inputs match within each preset's repeats; the same
frozen guard checks each intended first request and temperature zero before
provider forwarding. Later model choices remain outcomes. This is a preset with
its matching instruction, not tool removal alone, automatic batching or compaction.

The original T01 request, Duodoro source `5324cab`, visible command, protected
baseline, external criteria and reference stayed fixed. Only
`client/src/lib/format.ts` may change. Editable tests cannot redefine acceptance.
Both arms used local `gemma4:12b` at the frozen digest/Ollama 0.33.2, Python 3.14.5,
Node 24.22.0, compact/full-or-whole-file, temperature zero and thinking off. Each
started unloaded. Limits remain 300 dispatch seconds, 64,000 total reported / 8,192
output tokens, 24 governed tools, 12 iterations, 60-second requests, 2,048 request
output, 256 minimum output and context 16,384. Required verification, one completion
retry, plan gate off, root injection/support off retain the frozen profiles.
No dependency install, warm-up, model switch, paid call or prompt default change.

## Tool use, accounting and limits

Both full attempts read instructions, listed the directory, read source/tests and
planned, then used a rejected escaped-newline exact edit and one successful edit
of only the first formatter. They stopped before fixing `formatTime` or checking;
external acceptance rejects its non-finite behavior. Both whole-file attempts
followed the same read/plan tool-name sequence, wrote both fixes in one call, ran
the configured check and answered. Final file bytes match within each preset's
repeats. This describes observed paths, not deterministic future behavior.

Successful target `write_file` was observed in
0/2 full attempts and
2/2 whole-file attempts.
Availability or removal of a tool alone is not activation or recovery learning.
Full can also choose whole-file writes; overlap remains visible. Available normalized
archives and ordered tool records, including errors/rejections, remain local.
Suppressed/raw provider-wire calls are not reconstructed. Missing finalized
transcripts are unknown, not zero tool use or a fabricated explanation.

4/4 attempts have known full-response usage. Those known reports total
**220,580 prompt / 3,901 output tokens**. These are complete reported totals
for this cohort, not independently attested backend accounting. Durable budget rows preserve requested allowances,
heuristic/anchor estimates and final token stops. Whole-file used more reported
tokens to finish than the incomplete full attempts; that is not an efficiency
ranking at equal completed work. Provider reports and observed
upper bounds do not attest backend accounting, every dispatch or hard deadlines;
tool-call record counts can include rejected/undispatched calls.

Whole-file writes lack exact-edit stale-hash protection and can overwrite unrelated
content; larger files may need more generation. The fixed preservation/semantic
checks assess this task, not arbitrary repositories or external writers. Process-
local admission/path checks provide no host isolation or cross-process lock.
Temperature zero, two repeats, a fixed counterbalanced order and one backend do
not establish deterministic replication, held-out/category-level evidence or a
performance/quality ranking. Historical October 7 whole-file results remain mixed;
these selected-task outcomes do not overwrite them.

## Final-answer limitations

Both whole-file answers correctly report the observed passing check but also
include an unexecuted, inaccurate Python illustration labeled as implied test
content. It imports the TypeScript module as Python and misstates NaN's JavaScript
type. The actual unchanged Python test invokes Node; that illustration was neither
created nor run. Limited source/tool review records this issue separately in the
structured rows. Completed-and-accepted does not certify all answer claims and
these notes are not a complete semantic answer score.

## Audit, cleanup and disposition

Final offline external regrades reproduced every saved artifact grade. The audit
verified one durable run per attempt, matching saved archives/CLI exports, frozen
settings/criteria/reference/staging hashes, captured profiles/first identities,
clean pinned runtime/environment and protected-file preservation. Numeric journals,
bounded diffs/checkpoints, final artifacts, client/server logs and local operator
recipes are retained under
`~/.local/share/agent-harness/trials/duodoro-toolset-pl69m2cm/`.
The public result embeds the audit and recipe/log evidence hashes; it is not a
portable full-source/full-trace kit.

The owned API processes stopped and listeners are absent; evaluation-loaded Gemma
was unloaded after each attempt. The existing Ollama service remains running.
No Duodoro working-checkout edits/commits/pushes, paid calls, new model, Docker
action or manual artifact repair occurred. Execution used the reviewed inputs and
preconditions, with no coding intervention
or execution-setup correction. Local operator/report/audit recipes are archived;
postprocessing did not change artifacts, inference settings or submissions.

Keep `full` as the production default and retain the unchanged release criteria.
This one public duration task does not supply a ten-task useful envelope, real
partial-edit cancellation/reversal, held-out evidence or a second-model comparison.
The original ten-task no-go remains. Any follow-up needs its own explicit scope,
retention/activation criteria and comparable limits; do not infer release readiness
or silently launch replacement tasks from accepted patches alone.

This report changes docs/data only. Preparation's one shared pinned runtime passed
71 focused and 1,344 offline tests (five live deselected), static checks and both
scripted 90/90 matrices. Those are historical mechanics checks, not newly rerun
model-quality evidence. The new validation is regrading and auditing these four
saved real-tool observations.
