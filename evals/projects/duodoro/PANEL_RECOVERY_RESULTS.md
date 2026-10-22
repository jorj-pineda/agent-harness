# Live panel recovery — observed results, 2026-10-22 UTC

**The one frozen panel workflow passed its exercised recovery/preservation checks.**
Local Gemma wrote the formatter fix through native SSE, was explicitly cancelled,
then the panel reviewed saved work after an API restart and submitted one guarded
fork. The fork checked and answered; independent final acceptance passed. Panel
source-file reversal restored baseline target bytes/mode while preserving the
declared later line. Another restart exposed the saved outcome without replay.

This is one selected public request variant, not a model-quality/client comparison,
replacement for the ten-task trial or release approval. **Aggregate token usage is
unknown** because cancellation interrupted a request. The source patch was already
correct; the fork did not repair it. Original cohorts/no-go and broad G2–G5 remain
unchanged. The earlier [CLI observation](RECOVERY_RESULTS.md) and this panel result
cover one bounded workflow through each client, not general recovery reliability.

## Frozen runtime and both phases

[Protocol](PANEL_RECOVERY_PROTOCOL.md), [execution freeze](PANEL_EXECUTION_FREEZE.md),
[manifest](panel-execution-freeze.json) and
[structured observations](../../results/2026-10-22-duodoro-panel-recovery-workflow.json).
The clean application/UI runtime was `cd11e4c33fc89f4331b50891266a9c2fef57caef`.
Manifest SHA-256 remained
`2ae054141a3cbe889242b972c6aa099afcada8182af8d028705b9b4490f3b978`,
reviewed and merged in PR #170 before inference. Reporting commits were not the
execution runtime. Local `gemma4:12b` and its digest/Ollama version matched.

Fresh full Duodoro source/criteria/reference, actual root identity, settings,
managed environment and local recipes remained frozen. Browser control belonged
to the evaluation; owner `panel-q8dle5lu`, origin `http://127.0.0.1:18751` and absent
pending actions were rechecked before creating one capture-enabled standard
session. Scheduled stores were initially absent. Both phases reserved identical
150-second/32K-total-token/4,096-output/12-tool/six-iteration limits, compact/
whole-file tools and the exact required visible check. No budget transfer, third
turn, retry, provider pause, model switch, warm-up or manual patch repair occurred.

| Phase | Run | Completion / checks | Requests / responses | Reported usage |
|---|---|---|---:|---|
| Source | `67a352ec25c94923954d1057e1f1dac2` | cancelled / not_run | 5 / 4 | Unknown total; known prefix 9,604 prompt / 417 output |
| Guarded fork | `367aa986dd314d7da8c19fa3269cefbe` | completed / passed | 2 / 2 | 6,768 prompt / 301 output |

Source tools read the target and existing tests, emitted a plan and wrote the
target once. An initial 45-second observation wait expired while only reads/plan
existed: the hash was unchanged and no cancellation was clicked. The next
observation found a completed non-error target write card plus an independently
changed hash. **Cancel turn** was enabled and clicked once. The saved source
settled cancelled with checks not run and unknown token totals. Its first normalized
request and profile matched the frozen guard; no response was held or rewritten
to manufacture the window. Native EventSource SSE was used, with no POST fallback.

After owned settlement, API exit and model unload, the same settings/root/stores
restarted. Reload submitted nothing. Through panel controls, workspace review was
unchanged, transcript valid, configuration matching and context estimated to fit
(4,195 estimated prompt tokens, 2,048 request allowance). The latest source turn
was pinned, exact continuation/provider supplied and uncaptured-state risk
acknowledged before one **Fork and submit task**. Native guards accepted it.

Explicit **Check run** reads observed settlement; no automatic polling/retry was
introduced. The fork only ran `python -m pytest -q test_visible.py` and answered.
Visible output was `1 passed in 0.29s`. Its target before/after bytes were equal.
Source/fork profiles matched the freeze and the source archive remained unchanged.
The answer's formatter/check explanation agrees with observed tools/files. Its
“No other files” statement holds for eligible source files; ignored pytest/Python
cache entries were created and separately reported. This is limited answer review,
not a complete semantic-answer rubric or proof from prose alone.

The fork API/model stopped before scoring. Independent external acceptance passed
with untruncated output on a full retained final artifact, and independent regrading
reproduced it. Grading actual source after-capture target bytes also passed. Only
the allowed formatter differed from the eligible baseline; protected files/modes
were preserved. This demonstrates review/check/completion of interrupted correct
work, not correction of an incorrect patch.

## Reversal, restart and preservation

After scoring, the exact frozen later line was appended to existing
`preexisting-user-work.txt`. No task code was repaired.

- Fork reversal was **ineligible** because its target before/after state was equal;
  no no-op mutation was submitted.
- Selected the source run, opened its diff and the reversal disclosure, supplied
  the target/acknowledgment and clicked **Reverse selected file** once. Outcome
  `applied`, request `23dedc2e-2c7b-434d-bc43-3d41ec578a11`, journal
  `dddaaa19c5194381ac5e192b7b36fb64`.
- Kept that sole reversal key retained. After another actual API restart and page
  reload, **Inspect saved outcome** returned `applied`; history exposed both runs.
  There was no key forgetting or claim of arbitrary-key lookup coverage.
- Exactly two coding runs and one applied reversal exist. Read-only exports before
  and after the final restart match. Run records, saved session/response evidence
  and journals stayed unchanged; retirement remains explicit.
- A full eligible manifest found restored baseline bytes/modes everywhere except
  the one declared sentinel append. Initial sentinel content survived. Criterion,
  reference, runtime, settings and recipe hashes remained unchanged; canonical
  workspace inode/device identity was retained. Full scored and post-reversal
  artifacts are preserved separately. Saved passing checks remain historical
  after the reversal and are not claims about the restored original formatter.

All create/send/cancel/review/fork/reversal/lookup actions used rendered panel
controls. Read-only DOM/localStorage/file/SQLite observations recorded evidence;
operational helpers handled owned server lifecycle/model unload and independent
grading. No injected application calls/storage mutations or direct API/CLI
workflow workaround substituted for a browser control. HTTP observer counts
confirm one SSE submission, one cancellation, one continuation and one reversal,
with zero coding/file-action resubmissions on reload or lookup.

## Timing, cleanup and remaining limits

Saved source/fork lifecycle intervals were 72.35 and 60.79 seconds. These are not
pure inference/dispatch measurements or hard end-to-end bounds; setup, observation,
storage, review, restarts, grading and cleanup add time. Known response prefixes
sum to 16,372 prompt / 718 output tokens, **not** aggregate usage or proven total
token compliance. The cancelled fifth source request retains its unknown tail.

All four owned APIs exited, port 18751 was released, Gemma unloaded and the owned
browser tab closed. The existing Ollama service remains. No paid calls, additional
models, Docker actions or Duodoro working-checkout/index/branch edits occurred.
Local scoped plaintext logs, browser screenshots/observations, numeric journals,
archives, recipes, grader outputs and their hashes remain at the structured
report's evidence root. They are retained local evidence, not a downloadable bundle.

Ignored contents, ACLs/ownership, external writers and cross-process coordination
remain outside preservation guarantees. Checks use the Node library adapter, not
the full Next/Vitest suite or hostile-code isolation. This is not a held-out or
repeated quality experiment, causal recovery gain, unattended task envelope or
release approval. Original trial/comparison counts and broad G2–G5 stay unchanged.

This PR changes docs/data only. Freeze G0 (53 focused, 1,363 offline, static checks
and both scripted 90/90 matrices) remains historical preparation evidence; it was
not rerun for the reporting slice or presented as model-quality evidence.
