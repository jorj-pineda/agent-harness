# Four-cell buffered/streamed observations

After independent freeze re-review and PR #186's merge, all four original T09
attempts ran once through the actual CLI at runtime
`51463aedc97dee0a17ddc5cc032ffb7b8f4f4445`, in the reviewed B1/S1/S2/B2 order.
**Buffered: 0/2 accepted artifacts, 0/2 completed-and-accepted. Streamed: 0/2
accepted artifacts, 0/2 completed-and-accepted.** All four failed with provider
read timeouts before edits. Streaming did not avoid the observed timeout in this
selected comparison. This does not identify the backend cause or prove streaming
cannot help another task/configuration.

The [unchanged freeze](TRANSPORT_COMPARISON_FREEZE.md) and
[structured results](../../results/2026-10-23-duodoro-transport-comparison.json)
retain task/model/settings identities, every outcome, external grades/regrades,
request/run IDs, observations, preservation and cleanup. No replacement attempt,
operator coding rescue, prompt hint, timeout/budget increase, paid/new model,
dependency install, Docker action or Duodoro working-checkout edit occurred.

| Cell | Mode | Requests / responses / failures | Artifact | Fifth-call transport duration | Fifth-call received chunks | Known response-prefix prompt / output |
|---|---|---|---|---|---|---|
| B1 | buffered | 5 / 4 / 1 | rejected, unchanged | 60,012.784 ms | 0 | 13,536 / 132 |
| S1 | streamed | 5 / 4 / 1 | rejected, unchanged | 60,009.155 ms | 0 | 13,536 / 131 |
| S2 | streamed | 5 / 4 / 1 | rejected, unchanged | 60,005.407 ms | 0 | 13,536 / 131 |
| B2 | buffered | 5 / 4 / 1 | rejected, unchanged | 60,011.950 ms | 0 | 13,536 / 132 |

All four aggregate token totals remain **unknown**, including their failed
in-flight requests. Prefix usage is neither a total nor billing evidence or proof
of within-budget success. Failed saved runs have no final response envelope;
completion/check fields therefore remain unrecorded. Complete untruncated tool
progress records tree, read_file and emit_plan ending without errors, with no
write or command check. Each first normalized response is empty with zero tools;
the existing one-completion recovery is followed by those three tool responses.
No final answer or check outcome is reconstructed from partial progress.

## Transport activation and hypothesis

All four actual first normalized identities match the freeze. Recomputed first
wire hashes match; parsed bodies are identical after removing only `stream`.
The same literal workspace path, fresh baseline bytes/modes and separate unseeded
stores were used. Later generated histories/tool-call IDs may diverge; first-input
parity does not assert identical later requests or a causal model effect.

The treatment was activated: each streamed attempt returned four complete
terminal-bearing responses before the timeout. Its first response contained one
frame in one received body chunk; the next three each contained two frames in two
chunks. Both fifth streamed calls received **zero chunks and no terminal frame**
before timeout. Both buffered fifth calls also received zero chunks. The experiment
therefore provides no evidence that this treatment removed the specific failure.
No partial tool executed, and no failed payload can be reconstructed from these
numeric observations.

All **20/20 request events** have external transport observations and corresponding
saved response/failure observations, without budget-prefix truncation. Sixteen
returned responses have present native usage fields; the four failed calls retain
duration only with unknown terminal/content/tool-count/usage metadata. Client
received-chunk timing is not wire-packet timing, token arrival, queue time or
backend generation evidence. A failed tail is not a measured inter-chunk gap.

Experimental transport duration includes buffering, assembly and cleanup; the
existing adapter's HTTP-post duration includes that work here too. They are not
pure inference timing or parsing-excluded network measurements. The unchanged
60-second timeout bounds read inactivity; chunk delivery could permit a longer
streamed call, while the shared 300-second dispatch limit remains. No call here
established such a longer successful streamed generation. Source API lifecycles
lasted about 126–132 seconds each, including startup and shutdown, separate from
per-call duration and hard end-to-end deadlines. Parsing/byte/frame/tool bounds
remain part of treatment. The configured API profile does not attest this override;
the external manifest and recipe hashes identify it.

## Artifact, restart and cleanup audit

Each cell has exactly one durable run, one chat submission and no reversal.
Printed request keys were retained and inspected through actual CLI run lookup.
After settlement, complete source-API shutdown and model unload, each full artifact
was copied and independently graded twice without changing its bytes/modes. All
four rejected artifacts are unchanged; protected files/modes and original
criteria/reference/settings remain intact. The next common-path staging occurred
only after the prior archive and current workspace matched the audited full tree
and root identity. No user workspace was reset.

Each guarded review-only API restart returned identical CLI saved JSON; its HTTP
journal contains GETs only and no provider calls. The final read-only audit also
compares saved records to SQLite, revalidates all transport records, recomputes
first wire hashes and checks every archived full tree and bounded diff hash.
All 95 retained outcome/reporting/progress evidence hashes match. The raw local
results and committed preparation manifest remain unchanged, including the
manifest's deliberately unrun result fields. Reporting results do not rewrite it.

All eight owned APIs exited with complete shutdown logs; port 18758 is released.
Gemma was unloaded after each source phase and the final loaded-model list is empty.
The existing Ollama service remains running. Duodoro head/status/index/refs
fingerprints match their pre-trial values. All frozen recipes, runtime/source and
preflight evidence hashes match; historical cohorts and failed freezes are intact.

No execution correction or coding rescue was needed. A post-execution read-only
inspection queried a nonexistent `sessions.record` column; schema inspection then
used `sessions.archive` and `tool_progress.record`. This reporting correction
changed no state, artifact, criterion or model attempt. The preparation's earlier
audit-gate correction remains separately documented and retained before inference.

Immutable local evidence and recipes remain under:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-transport-comparison-2gxp1b38`

## Validation and disposition

The actual frozen launcher completed all four scheduled cells and its archive/
regrade/restart gates. A separate post-execution read-only audit passed identities,
coverage, preservation, SQLite equality and process/model cleanup. JSON/report
consistency and Git whitespace checks passed. This docs/data report does not rerun
or reinterpret the preparation's 100 focused/1,437 offline/static checks and two
scripted 90/90 matrices as task-quality evidence; they remain mechanics checks at
the exact execution runtime.

Do not promote streaming to an application default, raise budgets/timeouts, repair
these artifacts or repeat this cohort automatically. Two repeats per arm on one
selected public task and one model do not establish held-out/repeated two-model
improvement, an unattended envelope or release readiness. Original cohorts and
no-go/G2–G5 remain unchanged. A separate evidence-based decision must choose any
further bounded investigation or explicit narrower envelope.
