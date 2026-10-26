# Real-task baselines

Latest recovery observation: [one bounded live panel workflow](projects/duodoro/PANEL_RECOVERY_RESULTS.md)
at `cd11e4c` passed its exercised cancel/restart/review/fork/check/source-reversal
sequence on 2026-10-22 UTC, with independent acceptance/regrade. Source patch was
already correct; the fork checked/answered without edits. Cancelled usage is
unknown. This is a request-variant workflow observation, not a quality comparison
or replacement for historical cohorts; release no-go and broad G2–G5 stay unchanged.

Real coding evaluations: a live model uses the application runtime and real
tools on disposable copies of trusted fixtures in `evals/real_tasks/` and explicitly
authorized pinned tasks in `evals/projects/`.
Acceptance tests that the model cannot edit grade the final files. These results
say nothing about other models, tasks, or languages. Scripted-contract and
simulated-tool results live elsewhere and are not comparable.

Raw reports (traces, diffs, acceptance output, usage, runtime config) are in
`evals/results/`.

## 2026-10-26 — Original Duodoro T02/T08, Gemma / Beam descriptive cohort

The [results and limits](projects/duodoro/BEAM_COMPARISON_RESULTS.md) and
[raw sanitized rows](results/2026-10-26-gemma-beam-original-t02-t08-cli.json) record
eight actual CLI/shared-runtime attempts after independent protocol/freeze review
and CI/merge. Runtime `de6083d`, original public source/criteria, two repeats/model/
task and identical approved limits remain fixed. Gemma accepted completions: T02
0/2, T08 2/2 (2/4 total). Beam: T02 1/2, T08 0/2 (1/4 total). These are distinct
native model/provider configurations, not a causal or general ranking.

Both local timer calls time out on their second request with unknown aggregate
usage/absent TurnResponse. Hosted failures are truncation and two token-gate stops;
all failed artifacts remain initial. Successful patches pass external acceptance/
regrade; refactors also pass original source review. All 23 returned hosted responses
report zero Go cost, with backend/invoice limits retained. All eight original keyed
runs, archives/inputs/restart evidence and 24 owned API cleanup records are audited;
Gemma is unloaded and Duodoro unchanged. No repair/replacement/fallback. Original
cohorts/no-go, held-out targets and G2–G5 are unchanged.

## 2026-09-29 — `gemma4:12b`, server context 4,096 (baseline)

| Field | Value |
|---|---|
| Harness revision | `1cddce8` (clean). The raw report records `cb54ebe`: the same tree before its commit message was rewritten. |
| Model | `gemma4:12b` (Q4_K_M) via Ollama 0.33.2, local |
| Hardware | Apple M1 Pro, 16 GB unified memory |
| Server context | 4,096 tokens: the Ollama default in this setup. The harness did not set it. |
| Sampling | temperature 0.0 |
| Prompts | harness `coding-v2`; minimal `minimal-v1` (one sentence) |
| Attempts | 5 tasks × 2 modes × 3 repeats = 30; mode order alternates per task/repeat |
| Budgets | 8 model iterations, 24 tool calls, 600 s wall deadline, 300 s request timeout, no token budgets, verification not required, 1 completion retry |
| Raw report | `results/2026-09-29-gemma4-12b-baseline-ctx4k.json` |

```sh
OLLAMA_MODEL=gemma4:12b MAX_TURN_WALL_SECONDS=600 REQUEST_TIMEOUT_SECONDS=300 \
  uv run python -m evals.real_run --provider ollama --mode both --repeats 3 \
  --report /tmp/agent-harness-baseline-gemma4-12b.json
```

| Task | harness | minimal |
|---|---|---|
| decimal_total | 0/3 | 0/3 |
| divide_zero | 3/3 | 3/3 |
| parse_flags | 0/3 | 0/3 |
| slugify | 0/3 | 0/3 |
| stable_dedupe | 1/3 | 3/3 |
| **total** | **4/15** | **6/15** |

| Measure | harness | minimal |
|---|---|---|
| Terminations | 4 accepted, 11 truncated | 6 accepted, 8 truncated, 1 acceptance failed |
| False completion (claimed done, acceptance failed) | 0 | 1 |
| Median latency | 178 s | 191 s |
| Median tool calls | 3 | 3 |
| Median reported tokens (prompt + output) | 10,328 | 9,538 |

### Why attempts failed

| Failure | Attempts | Label |
|---|---|---|
| After `grep_repo` and `read_file`, the model produced about 2,200–3,500 output tokens with no visible content or tool call, and Ollama stopped at the length limit. The prompt at that step is about 1,900–2,000 tokens, so output ran until the 4,096-token window was full. This is consistent with hidden reasoning filling the window. It is not verified, because the runner does not store the raw provider payload. | 17 (9 harness, 8 minimal) | environment/setup: server context |
| `replace_text` received `old_text` containing a literal backslash-n (`\n` as two characters) instead of newlines, so no match was found. The model then wrote a long explanation that hit the length limit. | 2 (harness) | edit mechanics / tool interface |
| An incorrect rounding fix was declared complete without running a check. Verification was not required in this configuration. | 1 (minimal) | model reasoning + premature completion |

Other tool errors: `python3 -m pytest` was rejected by the command allowlist
twice (only `python`/`pytest` are allowed), and `write_file` was called once
without `path`. One harness attempt recovered from the `replace_text` failure by
re-reading the file and using `write_file`, and was accepted.

### Interpretation

- The 4/15 versus 6/15 difference is not evidence for either prompt. Three
  repeats at temperature 0 are few, and most failures come from the server
  context setting, not from prompt content.
- The harness prompt adds roughly 650 prompt tokens per request. Under a
  4,096-token window, that leaves less room for output.
- The dominant obstacle is environmental and removable: the harness never told
  Ollama what context length to use. The first experiment changes only that.

## 2026-09-29 — `gemma4:12b`, server context 16,384 (stopped early)

The only change from the baseline was `OLLAMA_NUM_CTX=16384`; Ollama confirmed
a 16,384-token context at the same 7.5 GiB footprint. The run was stopped after
8 of 30 attempts because the failure had not gone away, only moved: the three
tasks that had truncated at 4K now ended in `ProviderError` after about
330–390 s, when the 300 s request timeout fired mid-generation. The runner wrote
reports only at the end at the time, so just the progress log survives
(`results/2026-09-29-gemma4-12b-ctx16k-partial.log`). No counts from this run are
comparable.

A direct probe then replayed the `slugify` step after `read_file` with output
capped at 1,500 tokens. With the model default, the response carried about
2,000 characters in Ollama's `thinking` field (571 output tokens, 61 s), which
the provider discards. With `think: false`, the model called a tool after 77
output tokens (22 s). This confirms that the empty output in the baseline was
hidden reasoning.

## 2026-09-30 — `gemma4:12b`, thinking off, server context 4,096

The only change from the baseline was `OLLAMA_THINK=false`. Everything else
matches: model, machine, server context 4,096, temperature 0.0, prompts, budgets,
and task order.

| Field | Value |
|---|---|
| Harness revision | `53199ba` (clean). This is the same code as this PR's `OLLAMA_THINK` and per-attempt report commits, before they were rebased onto `main`. |
| Raw report | `results/2026-09-30-gemma4-12b-think-off-ctx4k.json` |

```sh
OLLAMA_MODEL=gemma4:12b OLLAMA_THINK=false MAX_TURN_WALL_SECONDS=600 \
  REQUEST_TIMEOUT_SECONDS=300 uv run python -m evals.real_run --provider ollama \
  --mode both --repeats 3 --report /tmp/agent-harness-exp-think-off.json
```

| Task | harness (baseline → thinking off) | minimal (baseline → thinking off) |
|---|---|---|
| decimal_total | 0/3 → 0/3 | 0/3 → 0/3 |
| divide_zero | 3/3 → 3/3 | 3/3 → 3/3 |
| parse_flags | 0/3 → 3/3 | 0/3 → 3/3 |
| slugify | 0/3 → 3/3 | 0/3 → 0/3 |
| stable_dedupe | 1/3 → 3/3 | 3/3 → 3/3 |
| **total** | **4/15 → 12/15** | **6/15 → 9/15** |

| Measure | harness | minimal |
|---|---|---|
| Terminations | 7 accepted, 3 iteration limit, 2 truncated (all 5 of those still passed acceptance), 3 iteration limit and failed | 9 accepted, 3 acceptance failed, 3 iteration limit and failed |
| False completion | 0 | 3 (`slugify`: incorrect fix reported as done) |
| Median latency | 63 s (baseline 178 s) | 55 s (baseline 191 s) |
| Median reported tokens | 13,662 | 9,880 |
| Verification | the model never ran a passing check (0/30) | same |

### What the result shows

- **Repeats were nearly identical.** At temperature 0 without thinking, the
  three attempts per task and mode mostly produced the same trace. Treat each
  cell as roughly one sample, not three independent ones. The harness versus
  minimal difference (12 versus 9) is one task, `slugify`, and is not evidence
  that either prompt is better.
- **Thinking was the main obstacle in the baseline.** Every
  task-and-mode cell except `decimal_total` improved or held. Latency fell by
  about two thirds. This measures one model's configuration; it does not show
  that thinking hurts coding in general.
- **Some turns did not end cleanly but still left a correct artifact.** Five
  harness attempts passed acceptance while ending at the iteration limit or
  with a truncated final answer. The runtime reported them as not completed,
  which is conservative, not a false claim.

### Why attempts failed or wasted steps

| Observation | Attempts | Label |
|---|---|---|
| `decimal_total`: two `write_file` edits, then two `python3 -m pytest` calls rejected by the allowlist, then the 8-iteration limit. The final code fails acceptance. | 6 (all) | tool interface + model reasoning |
| `run_command` with `python3` rejected (only `python -m pytest` or `pytest` are allowed). Every check the model attempted in the run was `python3`. | 18 calls in 8 attempts | tool interface |
| `replace_text` found no match: the model sent a literal backslash-n (`\n` as two characters) instead of newlines, as in the baseline. It recovered with `write_file` every time. | 9 calls in 8 attempts | tool interface / edit mechanics |
| `slugify` minimal: an incorrect rewrite was declared done without a check. | 3 | model reasoning + premature completion |

The next experiment, one variable again, makes these two tool errors
actionable: it names the escaped-newline mismatch and tells the model to use
`python` instead of `python3`. It does not rewrite the model's arguments.

## 2026-10-04 — `gemma4:12b`, clearer tool errors, thinking off, context 4,096

This experiment changes the tool-error feedback from the thinking-off run:
exact replacement errors suggest a unique match or copying the latest file
content, identify literal backslash-n characters when converting them to line
breaks would yield a unique match, and rejected `python3` commands recommend
`python`. The tools still reject invalid calls without rewriting arguments or
changing command permissions. The newline hint does not diagnose tab-only
mismatches or change literal escapes that already match source text.

| Field | Value |
|---|---|
| Harness revision | `a78f6e6` (clean) |
| Model/server/hardware | Same local `gemma4:12b` Q4_K_M, Ollama 0.33.2, Apple M1 Pro, 16 GB |
| Settings | Thinking off; server confirmed context 4,096; temperature 0.0 |
| Comparison | All recorded runtime settings and fixture source hashes match the thinking-off run above; same prompts, budgets, repeats, and alternating mode order |
| Attempts | 5 tasks × 2 modes × 3 repeats = 30, all included |
| Raw report | `results/2026-10-04-gemma4-12b-tool-errors-ctx4k.json` |

```sh
OLLAMA_MODEL=gemma4:12b OLLAMA_THINK=false MAX_TURN_WALL_SECONDS=600 \
  REQUEST_TIMEOUT_SECONDS=300 uv run python -m evals.real_run --provider ollama \
  --mode both --repeats 3 --report /tmp/agent-harness-tool-errors.json
```

| Task | harness (thinking off → clearer errors) | minimal (thinking off → clearer errors) |
|---|---|---|
| decimal_total | 0/3 → 0/3 | 0/3 → 0/3 |
| divide_zero | 3/3 → 3/3 | 3/3 → 3/3 |
| parse_flags | 3/3 → 2/3 | 3/3 → 3/3 |
| slugify | 3/3 → 3/3 | 0/3 → 0/3 |
| stable_dedupe | 3/3 → 3/3 | 3/3 → 3/3 |
| **total** | **12/15 → 11/15** | **9/15 → 9/15** |

| Measure | harness | minimal |
|---|---|---|
| Terminations | 7 accepted, 5 iteration limit, 1 truncated, 1 blocked, 1 acceptance failed | 9 accepted, 2 iteration limit, 4 acceptance failed |
| Completed status with failed acceptance | 1 | 4 |
| Attempts with a passing model-run check | 1/15 (previously 0/15) | 0/15 (unchanged) |
| Median latency | 87 s (previously 63 s) | 60 s (previously 55 s) |
| Median tool-call records | 7 (previously 5) | 4 (unchanged) |
| Median reported tokens | 18,733 (previously 13,662) | 9,527 (previously 9,880) |

Four harness artifacts passed acceptance despite an incomplete terminal state:
two `divide_zero` attempts reached the iteration limit, one `slugify` response
was truncated, and another was blocked by repeated unchanged calls. Acceptance
and clean completion remain distinct.

### Recovery and remaining failures

- **Newline hints:** 12 errors across nine attempts, versus nine errors across
  eight attempts previously. Two harness `divide_zero` attempts corrected the
  newlines and succeeded with `replace_text`. The other seven affected attempts
  recovered with `write_file`; six passed acceptance. All three harness
  `stable_dedupe` attempts repeated the newline error before falling back.
  There is limited observed recovery, but no reduction in edit-error calls.
- **Command hints:** 11 `python3` rejections, versus 18 previously. Only one
  attempt, harness `divide_zero` repeat 3, switched to `python -m pytest` and
  passed its visible test. It used the eighth iteration and still ended
  incomplete. Fewer rejections do not by themselves show better recovery:
  several attempts never tried a check. Other errors included one executable
  named `python -m pytest` and two rejected `python -c` calls.
- **Repeated-call blocking:** harness `slugify` repeat 3 emitted a batch of
  repeated `python -c` calls after editing. The runtime stopped on an unchanged
  repeated call and rejected the remaining 30 calls. Its 38 trace records
  include attempted calls that were not dispatched; they are not 38 executed
  tools. The final artifact passed acceptance.
- **`decimal_total`:** all six attempts failed independent acceptance. Five
  reached the iteration limit; minimal repeat 1 returned an empty answer with
  no tools, leaving the fixture unchanged. None executed a check successfully.
- **`parse_flags`:** harness repeat 3 stripped whole lines but failed to strip
  the individual key and value. It declared completion without a check and
  failed acceptance. This is the additional failed artifact versus the prior run.
- **Minimal `slugify`:** all three incorrect fixes were declared complete
  without checks, as before.

The raw report's `false_completion` field counts completed runtime status with
failed acceptance. Here the minimal count of four includes an empty answer,
so it must not be read as four explicit prose claims of success. Three minimal
`slugify` answers and one harness `parse_flags` answer did describe a completed
fix that failed acceptance.

### Interpretation and next experiment

Clearer feedback did **not** improve task acceptance in this comparison. Two
successful newline retries and one passing check are useful observations, not
evidence of a reliable improvement. The one-task decline in harness acceptance
also does not establish that the messages harmed performance.

Temperature 0 did not make all traces identical: some attempts diverged before
receiving any changed error feedback, including the empty minimal
`decimal_total` turn. These small runs on different days do not isolate latency
or token differences causally. Results remain limited to this model and five
fixtures. The diagnostics are retained as accurate, actionable tool feedback;
they do not add prompt instructions, permissions, or automatic argument repair.

A focused next experiment could remove the demonstrated interpreter-name
obstacle by accepting `python3 -m pytest` through the same controlled execution
path as `python -m pytest`, while retaining the existing module restriction.
That would be a separate implementation and comparison. Verification recovery
and the empty-answer completion case remain open; this experiment does not
justify a larger prompt or context-management framework.

## 2026-10-04 — `gemma4:12b`, Python pytest alias, thinking off, context 4,096

This experiment adds `python3 -m pytest` as an alias for the existing
PATH-resolved `python -m pytest` execution path. It does not select a separate
`python3` interpreter. Other modules, script files, and `-c` remain rejected.
Verification tracking recognizes the submitted alias; configured project checks
still require exact submitted argv. For example, a configured `python -m pytest`
check is not satisfied by submitting `python3 -m pytest`, despite the shared
execution path. Configure the spelling that the model is instructed to use.

Tool descriptions and allowed-command errors now advertise both spellings.
This comparison measures that advertised interface together with alias
execution, rather than isolating the execution change from its descriptions.
Tool trace arguments retain the submitted `python3` argv, while the result's
`argv` records the canonical `python` command that actually ran. Existing
response fields are preserved.

| Field | Value |
|---|---|
| Harness revision | `0ff51e6` (clean) |
| Model/server/hardware | Same local `gemma4:12b` Q4_K_M, Ollama 0.33.2, Apple M1 Pro, 16 GB |
| Settings | Thinking off; server confirmed context 4,096; temperature 0.0 |
| Comparison | All recorded runtime settings and fixture source hashes match the clearer-errors run; same system prompts, budgets, three repeats, and alternating mode order |
| Attempts | 5 tasks × 2 modes × 3 repeats = 30, all included |
| Raw report | `results/2026-10-04-gemma4-12b-python3-alias-ctx4k.json` |

```sh
OLLAMA_MODEL=gemma4:12b OLLAMA_THINK=false MAX_TURN_WALL_SECONDS=600 \
  REQUEST_TIMEOUT_SECONDS=300 uv run python -m evals.real_run --provider ollama \
  --mode both --repeats 3 --report /tmp/agent-harness-python3-alias.json
```

| Task | harness (clearer errors → alias) | minimal (clearer errors → alias) |
|---|---|---|
| decimal_total | 0/3 → 0/3 | 0/3 → 0/3 |
| divide_zero | 3/3 → 0/3 | 3/3 → 3/3 |
| parse_flags | 2/3 → 3/3 | 3/3 → 3/3 |
| slugify | 3/3 → 2/3 | 0/3 → 0/3 |
| stable_dedupe | 3/3 → 3/3 | 3/3 → 3/3 |
| **total** | **11/15 → 8/15** | **9/15 → 9/15** |

| Measure | harness | minimal |
|---|---|---|
| Terminations | 8 accepted, 4 acceptance failed, 3 blocked | 9 accepted, 4 acceptance failed, 2 blocked |
| Completed status with failed acceptance | 4 (previously 1) | 4 (unchanged) |
| Attempts with a passing model-run check | 3/15 (previously 1/15) | 3/15 (previously 0/15) |
| Final verification statuses | 3 passed, 12 not run | 1 passed, 2 stale, 12 not run |
| Median latency | 80 s (previously 87 s) | 56 s (previously 60 s) |
| Median tool-call records | 7 (unchanged) | 4 (unchanged) |
| Median reported tokens | 16,682 (previously 18,733) | 9,577 (previously 9,527) |

### What changed and what remains

- **Alias execution worked:** the model submitted six `python3 -m pytest
  test_visible.py` calls, one in every `decimal_total` attempt. All six executed
  through `python` and passed. No command call was rejected. Previously only
  one check executed and passed across the entire run.
- **Passing checks did not establish a correct patch:** all six decimal-total
  artifacts failed the independent empty-input check because the model used
  `sum(Decimal(price) for price in prices)` without a Decimal starting value.
  The resulting integer zero has no `quantize` method. Four attempts finished
  with passed verification and failed acceptance. Minimal repeats 2 and 3 edited
  after their checks, then hit repeated-write blocking; verification was stale.
  Some attempts added visible tests, but those did not cover empty input and
  cannot change authoritative acceptance.
- **Harness `divide_zero` declined:** all three attempts repeatedly submitted
  escaped-newline edits and were blocked before changing the fixture or calling
  a command. The previous run accepted all three artifacts. This is an observed
  regression for this configuration; the traces do not isolate why the model's
  edit behavior changed before any alias invocation.
- **Harness `slugify` declined:** repeat 3 deleted an underscore rather than
  treating it as a separator; `One_two` became `onetwo`. Minimal mode's three
  incorrect artifacts also remained unaccepted. None ran a check.
- **Edit errors persisted:** 17 escaped-newline mismatch errors and five
  repeated-call blocks were recorded. Three blocks involved exact replacements
  in harness `divide_zero`; two involved whole-file writes in minimal
  `decimal_total`. These attempted calls include calls rejected before dispatch.
- **Empty final answers remain a completion problem:** all six `slugify`
  attempts returned empty final answers while receiving completed runtime
  status. Four of those artifacts failed acceptance. Consequently the report's
  `false_completion` total of eight includes four empty answers, alongside four
  decimal-total completion claims. It is a status/acceptance proxy, not a count
  of eight explicit prose claims.

### Interpretation and next slice

The alias removes a demonstrated command-interface obstacle and enables real
checks under the existing execution restrictions. It did **not** improve task
acceptance: harness acceptance was worse and minimal acceptance unchanged.
It is retained for basic tool usability, not as a measured coding-quality gain.

This is still one model and five fixtures with nearly deterministic repeats.
Changed tool descriptions, fresh workspace paths, and variation seen in earlier
runs limit causal attribution; latency differences are descriptive. Do not assume
that accepting more valid commands makes the model's patches more correct.

The next focused work should address an observed failure: empty final-answer
completion or repeated edit mismatches, with bounded recovery and a separate
comparison. A smaller-toolset experiment is also supported by the repeated
exact-edit failures and successful whole-file fallbacks. None of that work is
implemented here, and adding a larger prompt or a model judge is not justified
by this run. The personal-use persistence, cancellation, and diff workflow
remains open.

## 2026-10-05 — `gemma4:12b`, blank-final recovery, thinking off, context 4,096

A normal final reply with no tool calls and empty or whitespace-only content now
triggers the existing bounded completion recovery. Feedback asks for a non-empty
answer with the observed result and unresolved problems. Missing verification
or blocked edits use the same retry counter, with combined feedback when needed.
After retries are exhausted, the runtime returns incomplete with an explicit
fallback instead of marking the blank reply completed. Tool-bearing replies may
still have empty content. Provider truncation and budget stops retain precedence;
recovery uses the usual pre-request and tool-dispatch gates.

This changes the shared application/evaluation loop. System prompts, tool
schemas, fixtures, acceptance checks, and recorded runtime settings are unchanged
from the Python alias comparison. The new feedback is injected only after a
blank final reply. A non-empty answer is not proof that the patch works.

| Field | Value |
|---|---|
| Harness revision | `35a2b59` (clean) |
| Model/server/hardware | Same local `gemma4:12b` Q4_K_M, Ollama 0.33.2, Apple M1 Pro, 16 GB |
| Settings | Thinking off; server confirmed context 4,096; temperature 0.0 |
| Comparison | Runtime settings and fixture source hashes match the Python alias run; eight iterations, 24 tool attempts, one completion retry, 600 s wall dispatch deadline, 300 s request timeout; token gates and required verification disabled |
| Attempts | 5 tasks × 2 modes × 3 repeats = 30, all included; alternating mode order |
| Raw report | `results/2026-10-05-gemma4-12b-empty-final-ctx4k.json` |

```sh
OLLAMA_MODEL=gemma4:12b OLLAMA_THINK=false MAX_TURN_WALL_SECONDS=600 \
  REQUEST_TIMEOUT_SECONDS=300 uv run python -m evals.real_run --provider ollama \
  --mode both --repeats 3 --report /tmp/agent-harness-empty-final.json
```

| Task | harness (alias → blank recovery) | minimal (alias → blank recovery) |
|---|---|---|
| decimal_total | 0/3 → 0/3 | 0/3 → 0/3 |
| divide_zero | 0/3 → 1/3 | 3/3 → 3/3 |
| parse_flags | 3/3 → 3/3 | 3/3 → 2/3 |
| slugify | 2/3 → 3/3 | 0/3 → 0/3 |
| stable_dedupe | 3/3 → 3/3 | 3/3 → 3/3 |
| **total** | **8/15 → 10/15** | **9/15 → 8/15** |

| Measure | harness | minimal |
|---|---|---|
| Runtime statuses | 6 completed, 7 budget exhausted, 2 blocked | 11 completed, 3 budget exhausted, 1 incomplete |
| Terminations | 3 accepted, 3 acceptance failed, 7 iteration limit, 2 blocked | 8 accepted, 3 acceptance failed, 3 iteration limit, 1 incomplete |
| Blank final answers marked completed | 0 (previously 3) | 0 (previously 3) |
| Completed status with failed acceptance | 3 (previously 4) | 3 (previously 4) |
| Attempts with a passing model-run check | 3/15 (unchanged) | 3/15 (unchanged) |
| Final verification statuses | 3 passed, 7 failed, 5 not run | 3 passed, 3 failed, 9 not run |
| Median latency | 79 s (previously 80 s) | 53 s (previously 56 s) |
| Median tool-call records | 7 (unchanged) | 4 (unchanged) |
| Median reported tokens | 18,211 (previously 16,682) | 9,572 (previously 9,577) |

### Observations and limitations

- **Blank completion is prevented:** none of the 30 attempts received completed
  status with blank text. Minimal `parse_flags` repeat 2 exhausted blank-final
  recovery and returned the explicit incomplete fallback. It had only inspected
  files, left the fixture unchanged, and failed acceptance.
- **Recovery can lead to unhelpful tool work:** all six `slugify` turns and all
  three harness `stable_dedupe` turns reached the iteration limit. Their traces
  include rejected direct Python execution and pytest against a source module,
  which collected no tests and exited with code 5. These are failed check
  attempts, not verification. Across the full run there were 18 rejected
  `python3 -c` calls. The previous run's six slug turns finished with blank text.
- **Artifact acceptance and completion differ:** all three harness slug patches,
  all three harness dedupe patches, and harness divide repeat 3 passed external
  acceptance despite reaching the iteration limit. The report's termination
  label records the runtime stop first; the primary acceptance count includes
  these seven artifacts. Minimal slug patches still failed separator handling.
- **Decimal correctness remains unresolved:** all six attempts ran passing
  visible tests and received completed status, but failed the independent
  empty-input case. `sum(Decimal(price) for price in prices)` returns integer
  zero on an empty list, so the subsequent `quantize` call fails. The report's
  six `false_completion` entries are these completed/failed-acceptance attempts;
  unlike the preceding run, none involve empty final answers. This guard does
  not prevent incorrect success claims in non-empty text.
- **Exact-edit failures persist:** 12 escaped-newline mismatch errors and two
  repeated-call blocks occurred, all in harness divide attempts. Two artifacts
  were unchanged; repeat 3 fell back to a correct whole-file write. Its later
  check collected no tests, and the turn reached the iteration limit.

The raw report stores terminal answers and tool traces, but not each model
response or injected recovery message. It cannot establish the exact number of
successful blank-answer recoveries or locate every changed decision relative to
feedback. Scripted tests establish the shared retry and budget contracts; the
live report establishes final statuses and independently scored artifacts.

Acceptance improved by two harness attempts and declined by one minimal attempt.
These small, nearly deterministic repeats on five fixtures do not establish a
reliable coding-quality gain. Fresh workspace paths and variation before changed
feedback limit causal attribution; latency differences are descriptive. The guard
is retained for honest completion bookkeeping, with extra tool work and iteration
exhaustion documented as costs. No prompt expansion or model judge is justified.

A next focused comparison could reduce exact-edit/tool-interface friction with a
smaller toolset or targeted mismatch recovery. Verification guidance and the
personal-use persistence, cancellation, and diff workflow remain open. Ollama
was unloaded and the server started for this experiment was stopped afterward.


## 2026-10-07 — `gemma4:12b`, whole-file toolset, thinking off, context 4,096

The opt-in `CODING_TOOLSET=whole_file` experiment removes only `replace_text`
from the shared application/evaluation registry (15 tools → 14). Retained
schemas and descriptions are unchanged. The harness prompt replaces its one
exact-edit instruction with a full-content `write_file` instruction, versioned
as `coding-v2-whole-file-v1`. Minimal mode retains `minimal-v1`. This measures
that tool reduction with the necessary matching instruction, not tool removal
in isolation. Memory tools, the semantic-search stub, planning, execution,
completion recovery, and permissions are unchanged. `full` remains the default.

The reduction was chosen from observed failures: the preceding run had 12
escaped-newline mismatch errors and two repeated-call blocks in harness
`divide_zero`; its only accepted harness division artifact used a whole-file
fallback. This experiment does not automatically repair tool arguments.

| Field | Value |
|---|---|
| Harness revision | `c331f65` (clean) |
| Model/server/hardware | Same local `gemma4:12b` Q4_K_M, Ollama 0.33.2, Apple M1 Pro, 16 GB |
| Settings | Thinking off; server confirmed context 4,096; temperature 0.0 |
| Comparison | All prior recorded runtime settings and fixture hashes match the blank-final run; eight iterations, 24 tool attempts, one completion retry, 600 s wall dispatch deadline, 300 s request timeout; token gates, required planning, and required verification disabled |
| Attempts | 5 tasks × 2 modes × 3 repeats = 30; all included, alternating mode order |
| Raw report | `results/2026-10-07-gemma4-12b-whole-file-ctx4k.json` |

The run loaded the prior report's non-secret runtime settings into isolated
`Settings(_env_file=None)` to avoid local `.env` drift:

```sh
uv run python - <<'PY'
import json
from evals import real_run
from harness.config import Settings

with open("evals/results/2026-10-05-gemma4-12b-empty-final-ctx4k.json") as f:
    config = json.load(f)["results"][0]["runtime_config"]
settings = Settings(
    _env_file=None, ollama_model="gemma4:12b", coding_toolset="whole_file", **config
)
real_run.get_settings = lambda: settings
real_run.main([
    "--provider", "ollama", "--mode", "both", "--repeats", "3",
    "--report", "/tmp/agent-harness-whole-file.json",
])
PY
```

For ordinary application/evaluator use, set `CODING_TOOLSET=whole_file` in the
normal settings environment. The real report now records the selected toolset
and full exposed tool specs; older reports lack those fields and represent the
historical full registry.

| Task | harness (blank recovery → whole-file) | minimal (blank recovery → whole-file) |
|---|---|---|
| decimal_total | 0/3 → 0/3 | 0/3 → 0/3 |
| divide_zero | 1/3 → 3/3 | 3/3 → 3/3 |
| parse_flags | 3/3 → 3/3 | 2/3 → 0/3 |
| slugify | 3/3 → 0/3 | 0/3 → 2/3 |
| stable_dedupe | 3/3 → 3/3 | 3/3 → 3/3 |
| **total** | **10/15 → 9/15** | **8/15 → 8/15** |

| Measure | harness | minimal |
|---|---|---|
| Runtime statuses | 8 completed, 7 budget exhausted | 12 completed, 3 incomplete |
| Terminations | 4 accepted, 4 acceptance failed, 7 iteration limit | 8 accepted, 4 acceptance failed, 3 incomplete |
| Completed status with failed acceptance | 4 (previously 3) | 4 (previously 3) |
| Blank final answers marked completed | 0 | 0 |
| Attempts with passing model-run checks | 3/15 | 3/15 |
| Final verification statuses | 3 passed, 5 failed, 7 not run | 3 passed, 12 not run |
| Median latency | 61 s (previously 79 s) | 46 s (previously 53 s) |
| Median tool-call records | 7 (unchanged) | 4 (unchanged) |
| Median reported tokens | 15,684 (previously 18,211) | 8,203 (previously 9,572) |

### Observations and decision

- **Division edit friction disappeared:** all six division artifacts passed.
  No exact-edit calls, mismatch errors, or repeated-call blocks occurred in the
  full run. Removing the tool removes that error opportunity; it does not prove
  the model learned to recover. All three harness division turns still reached
  the iteration limit after unsuitable checks.
- **Other task behavior worsened or varied:** all three harness slug rewrites
  deleted separators before converting them, failing acceptance. Two ended at
  the iteration limit after four invalid multiline command calls each. Minimal
  slug passed twice but failed separator handling once. All three minimal flags
  turns only read files, exhausted blank-answer recovery, and left the fixture
  unchanged. The traces do not isolate why these choices changed.
- **Visible checks still missed correctness:** all six decimal artifacts passed
  visible tests, including model-added cases, but failed independent empty-input
  acceptance and received completed status. These six and two completed slug
  failures account for the eight `false_completion` records. Editable tests
  cannot redefine authoritative acceptance.
- **Verification trouble remains:** 12 `python3 -c` rejections and eight invalid
  multiline argv calls occurred. Passing model-run checks occurred only in the
  six failed decimal attempts. Five accepted harness artifacts had incomplete
  turns: three division and two dedupe artifacts.

This small historical comparison does **not** establish a reliable coding-quality
improvement. Harness acceptance declined by one and minimal acceptance stayed
flat, hiding opposite changes across tasks. Fresh workspace paths, variation at
temperature 0, changed tool-schema context, and the necessary prompt adaptation
limit causal attribution. Latency/token differences are descriptive. The report
still stores terminal answers and tool traces, not every model response or
recovery message.

Keep `full` as the application default. Retain the opt-in path for reproducible
comparison, without recommending whole-file writes as a general improvement:
larger files require more generation and lose exact-edit stale-hash protection.
Verification guidance, bounded targeted recovery, and the personal-use
persistence/cancellation/diff workflow remain open. No model judge, larger
prompt, additional model, or premium supervisor was added. Ollama was unloaded
and the evaluation-owned server stopped afterward.

Validation: 500 offline tests passed (five live tests deselected); Ruff and mypy
passed. Both scripted scenario matrices passed 90/90 escalation decisions and
retained `scripted-contract` labels. Those validate harness contracts, not model
coding quality.


## 2026-10-07 — `gemma4:12b`, configured-check guidance, thinking off, context 4,096

This results-only experiment exercises the existing shared `PROJECT_CHECK_ARGV`
setting before adding recovery code. The only recorded configuration change from
the whole-file comparison is `None` →
`["python3", "-m", "pytest", "test_visible.py"]`. The shared runtime appends its
existing one-sentence command guidance to both prompt modes and recognizes only
that exact submitted argv as relevant verification. Required verification stays
**disabled**, so this does not test enforcement or its completion-retry behavior.
The fixture's visible check is supplied explicitly; no project detection or
independent acceptance requirements are exposed to the model.

The observed motivation was 12 rejected direct-Python calls, eight invalid
multiline argv calls, and passing checks only in six incorrect decimal attempts
in the preceding run. Keep `whole_file` fixed for this comparison to isolate
configured guidance from the prior tool reduction. This does not promote that
experimental toolset or change the application's `full` default.

| Field | Value |
|---|---|
| Harness revision | `4371e7b` (clean, merged #37); runtime matches the prior experiment's implementation |
| Model/server/hardware | Same local `gemma4:12b` Q4_K_M, Ollama 0.33.2, Apple M1 Pro, 16 GB |
| Settings | Thinking off; server confirmed context 4,096; temperature 0.0; whole-file toolset |
| Comparison | Budgets, fixture hashes, exposed schemas, and all recorded settings except `project_check_argv` match the preceding comparison |
| Budgets | Eight iterations, 24 tool attempts, one completion retry, 600 s wall dispatch deadline, 300 s request timeout; token gates, required planning, and required verification disabled |
| Attempts | 5 tasks × 2 modes × 3 repeats = 30; all included, alternating mode order |
| Raw report | `results/2026-10-07-gemma4-12b-project-check-ctx4k.json` |

Base prompt versions remain `coding-v2-whole-file-v1` and `minimal-v1`;
`project_check_argv` records the additional shared system-message block. These
version labels alone do not imply identical final system messages.

The comparison loaded prior non-secret settings into isolated settings:

```sh
uv run python - <<'PY'
import json
from evals import real_run
from harness.config import Settings

with open("evals/results/2026-10-07-gemma4-12b-whole-file-ctx4k.json") as f:
    config = json.load(f)["results"][0]["runtime_config"]
config.pop("tool_specs")  # Report evidence, not a setting.
config["project_check_argv"] = ["python3", "-m", "pytest", "test_visible.py"]
settings = Settings(_env_file=None, ollama_model="gemma4:12b", **config)
real_run.get_settings = lambda: settings
real_run.main([
    "--provider", "ollama", "--mode", "both", "--repeats", "3",
    "--report", "/tmp/agent-harness-project-check.json",
])
PY
```

For ordinary application use, set `PROJECT_CHECK_ARGV` to an explicit command
appropriate to the repository. `test_visible.py` is specific to these fixtures.
The application and real evaluator already share this guidance path.

| Task | harness (unguided → guided) | minimal (unguided → guided) |
|---|---|---|
| decimal_total | 0/3 → 0/3 | 0/3 → 0/3 |
| divide_zero | 3/3 → 3/3 | 3/3 → 3/3 |
| parse_flags | 3/3 → 3/3 | 0/3 → 0/3 |
| slugify | 0/3 → 0/3 | 2/3 → 0/3 |
| stable_dedupe | 3/3 → 3/3 | 3/3 → 0/3 |
| **total** | **9/15 → 9/15** | **8/15 → 3/15** |

| Measure | harness | minimal |
|---|---|---|
| Runtime statuses | 6 completed, 8 budget exhausted, 1 incomplete | 4 completed, 3 budget exhausted, 5 blocked, 3 incomplete |
| Terminations | 6 accepted, 8 iteration limit, 1 incomplete | 3 accepted, 1 acceptance failed, 3 iteration limit, 5 blocked, 3 incomplete |
| Completed status with failed acceptance | 0 (previously 4) | 1 (previously 4) |
| Attempts with a passing configured check | 11/15 (previously 3/15 with any passing check) | 7/15 (previously 3/15 with any passing check) |
| Final verification statuses | 9 passed, 4 stale, 1 failed, 1 not run | 6 passed, 1 stale, 8 not run |
| Accepted artifacts with incomplete turns | 3 (previously 5) | 0 (unchanged) |
| Median latency | 59 s (previously 61 s) | 43 s (previously 46 s) |
| Median tool-call records | 7 (unchanged) | 6 (previously 4) |
| Median reported tokens | 16,880 (previously 15,684) | 13,701 (previously 8,203) |

### Observations and decision

- **Valid checks became more common:** 25 command calls executed, all with the
  configured argv (18 harness, seven minimal). Direct `python3 -c` and multiline
  argv rejections disappeared. Harness dedupe and flags all finished cleanly with
  passing checks; all three harness division artifacts passed but still reached
  the iteration limit despite passing checks.
- **Minimal mode introduced a different interface failure:** all three dedupe
  turns and slug repeats 2/3 repeatedly supplied `argv` as a JSON string instead
  of a list, hitting the existing repeated-call gate before any edit. Some also
  guessed `test_dedupe.py` or `test_slug.py` rather than the supplied filename.
  Those five unchanged fixtures failed acceptance. Across both modes there were
  16 argument-validation errors, five repeated-call rejections, and three command
  names containing Markdown punctuation rejected by the allowlist. These are
  recorded errors, not automatic argument repair or executed commands.
- **Checks still missed or failed to repair defects:** minimal slug repeat 1
  passed the visible check, declared completion, and failed independent underscore
  acceptance. Harness slug repeat 1 added an underscore test, ran it, and ended
  with a failed check without repairing the defect. Repeat 2 added an underscore
  test after its last check and did not rerun it, making verification stale.
  Harness repeat 3 exhausted blank recovery after an incorrect edit without a
  check. The added test cases are model choices, not authoritative acceptance.
- **Decimal correctness stayed unresolved:** all six artifacts failed acceptance
  and all six turns reached the iteration limit. Four failed on empty input;
  harness repeats 2/3 changed quantization to `Decimal("1.00")` and failed half-up
  rounding on `1.005` before reaching the empty-input case. Three harness turns
  and minimal repeat 2 had stale final verification. Lower completed/failed counts
  therefore do not establish safer or more correct patches.
- **Blank completion remains bounded:** all three minimal flags turns left the
  fixture unchanged after reads and returned incomplete. No blank final answer
  received completed status in the full run.

Configured guidance increased successful check invocation but did **not** improve
artifact acceptance: harness stayed flat and minimal declined by five. It also
coincided with more incomplete turns (20/30 versus 10/30), so the drop in the
report's `false_completion` proxy from eight to one is not a correctness gain.
This small historical comparison, fresh disposable paths, and variation at
temperature 0 limit causal attribution. Latency differences are descriptive.
It does not measure the `full` toolset, required-verification enforcement,
other repositories, or other models. Terminal answers and tool traces still do
not include every model response or injected recovery message.

Keep the existing explicit project-check setting available for repository-specific
use; change no default, tool permission, runtime prompt template, or enforcement
policy on the strength of this run. Targeted feedback for malformed command
arguments and bounded recovery after failed checks are supported candidates for
a future focused slice. Personal-use persistence, cancellation, and diffs remain
open. The evaluation-owned model was unloaded and server stopped afterward.

Validation: the runner validated initial fixtures fail and reference solutions
pass before inference; all 30 result settings/schemas/fixture hashes were audited
against the prior report. Twelve focused configuration/verification tests passed.
Both scripted scenario matrices passed 90/90 escalation decisions and retained
`scripted-contract` labels, which validate harness mechanics, not model quality.
No implementation or evaluation-runner behavior changed, so the full offline
gate was not rerun for this documentation/results-only PR. The preceding
implementation's 500-test/Ruff/mypy gate remains historical evidence.


## 2026-10-07 — `gemma4:12b`, string-argv feedback, thinking off, context 4,096

`RunCommandInput` now rejects a string-valued `argv` with targeted feedback:
resend a JSON array of separate strings rather than a quoted array or shell
command, with an executable/module example, and use the configured project's
exact check tokens when supplied. The Pydantic error is returned through the
existing registry/loop path. It does not parse JSON strings, split shell text,
rewrite arguments, or dispatch invalid input. Valid arrays, schemas, tool
permissions, system prompts, and all budgets remain unchanged. Corrections must
come from the model within the existing iteration/tool/repeated-call gates.

The preceding guided run had 16 string-argv validation errors and five blocked
turns before edits. This experiment changes only that error feedback. It leaves
whole-file editing and explicit fixture-check guidance enabled for comparability;
it does not establish either as an application default or test required
verification enforcement.

| Field | Value |
|---|---|
| Harness revision | `58b95f6` (clean) |
| Model/server/hardware | Same local `gemma4:12b` Q4_K_M, Ollama 0.33.2, Apple M1 Pro, 16 GB |
| Settings | Thinking off; server confirmed context 4,096; temperature 0.0; whole-file toolset; configured `python3 -m pytest test_visible.py` |
| Comparison | All recorded runtime settings, exposed schemas, fixture hashes, and base prompts match the configured-check comparison |
| Budgets | Eight iterations, 24 tool attempts, one completion retry, 600 s wall dispatch deadline, 300 s request timeout; token gates, required planning, and required verification disabled |
| Attempts | 5 tasks × 2 modes × 3 repeats = 30; all included, alternating mode order |
| Raw report | `results/2026-10-07-gemma4-12b-argv-feedback-ctx4k.json` |

Reproduce with prior non-secret settings and a unique report path:

```sh
uv run python - <<'PY'
import json
from evals import real_run
from harness.config import Settings

with open("evals/results/2026-10-07-gemma4-12b-project-check-ctx4k.json") as f:
    config = json.load(f)["results"][0]["runtime_config"]
config.pop("tool_specs")
settings = Settings(_env_file=None, ollama_model="gemma4:12b", **config)
real_run.get_settings = lambda: settings
real_run.main([
    "--provider", "ollama", "--mode", "both", "--repeats", "3",
    "--report", "/tmp/agent-harness-argv-feedback.json",
])
PY
```

| Task | harness (old errors → targeted feedback) | minimal (old errors → targeted feedback) |
|---|---|---|
| decimal_total | 0/3 → 0/3 | 0/3 → 0/3 |
| divide_zero | 3/3 → 3/3 | 3/3 → 3/3 |
| parse_flags | 3/3 → 3/3 | 0/3 → 0/3 |
| slugify | 0/3 → 0/3 | 0/3 → 0/3 |
| stable_dedupe | 3/3 → 3/3 | 0/3 → 2/3 |
| **total** | **9/15 → 9/15** | **3/15 → 5/15** |

| Measure | harness | minimal |
|---|---|---|
| Runtime statuses | 7 completed, 7 budget exhausted, 1 incomplete | 1 completed, 7 budget exhausted, 4 blocked, 3 incomplete |
| Terminations | 7 accepted, 7 iteration limit, 1 incomplete | 1 accepted, 7 iteration limit, 4 blocked, 3 incomplete |
| Completed status with failed acceptance | 0 (unchanged) | 0 (previously 1) |
| Attempts with a passing configured check | 13/15 (previously 11/15) | 7/15 (unchanged) |
| Final verification statuses | 11 passed, 2 stale, 1 failed, 1 not run | 5 passed, 2 stale, 8 not run |
| Accepted artifacts with incomplete turns | 2 (previously 3) | 4 (previously 0) |
| Median latency | 59 s (unchanged rounded) | 44 s (previously 43 s) |
| Median tool-call records | 7 (unchanged) | 7 (previously 6) |
| Median reported tokens | 16,859 (previously 16,880) | 14,957 (previously 13,701) |

### Observations and limitations

- **Later valid argv was observed:** 11 attempts contained string arguments;
  six later dispatched a valid array: all three minimal division attempts,
  minimal dedupe repeat 2, and harness decimal repeats 2/3. Some sequences first
  changed shell-command text into a still-invalid quoted JSON array. Harness
  decimal repeat 2 corrected the invocation, ran the initially failing fixture
  check, then edited without rerunning, ending with stale verification. A valid
  call is not equivalent to a passing check or a correct patch.
- **Errors did not decline:** 23 string-argv validation errors were recorded,
  versus 16 previously; all received the new instruction. Four repeated-call
  blocks occurred versus five, with three malformed executable-name rejections
  unchanged in count. All three minimal slug turns and minimal dedupe repeat 1
  were blocked and left fixtures unchanged. Minimal dedupe repeat 3 produced a
  correct artifact but never corrected its command arguments before the iteration
  limit. The feedback does not reliably break this failure loop.
- **The two new accepted dedupe artifacts were fixed before feedback:** repeats
  2/3 inspected the visible test and edited before their first string-argv error.
  In the prior report both stopped on malformed commands before editing. Their
  acceptance difference therefore cannot be attributed solely to the changed
  feedback. Repeat 2 subsequently used a valid array and finished cleanly;
  repeat 3 ended incomplete with no check. Minimal division also diverged before
  feedback, adding a visible-test read; all three artifacts passed, but the
  additional failed command attempts exhausted iterations where prior turns
  completed cleanly.
- **Correctness failures remain:** all six decimal artifacts failed independent
  empty-input acceptance and reached the iteration limit. Four final verification
  states were passed and two stale. All three harness slug artifacts still
  deleted underscores before separator conversion; two added visible tests but
  did not repair the defect. Minimal flags left all three fixtures unchanged and
  exhausted blank recovery. No blank final answer received completed status.

Harness acceptance stayed flat and minimal rose by two, with observed corrections
but more string-argument errors. These small historical comparisons on five
fixtures do **not** establish a reliable coding-quality gain. Fresh workspace
paths and observed decisions diverging before the changed feedback limit causal
attribution. The raw report records terminal answers and tool-call order, not
every model response or tool-batch boundary; later valid calls are observed
sequences, not proof that the hint caused them. Latency/token differences are
descriptive. Zero `false_completion` records reflect terminal status: 22/30
turns were incomplete versus 20/30 previously, and 16 artifacts still failed
acceptance. They do not prove correctness of all work or safer completion.

Retain the accurate diagnostic as basic tool usability, without claiming a
model-quality improvement or increasing execution permissions/retry budgets.
Failed-check recovery, complete response traces, and personal-use persistence,
cancellation, and diffs remain open. Any further recovery strategy needs a
separate bounded implementation and measurement; this run does not justify a
larger prompt, model judge, extra model, or premium supervisor. The evaluation-owned
model was unloaded and server stopped afterward.

Validation: 93 focused tests and 506 offline tests passed (five live tests
deselected); Ruff and mypy passed. New regressions prove string arguments do not
dispatch processes, valid list retries execute unchanged, and existing repeated-call
and tool-call limits still stop recovery. Both scripted matrices passed 90/90
escalation decisions with `scripted-contract` labels; these validate mechanics,
not model quality. All 30 recorded configs, schemas, and fixture hashes were
audited against the prior report; initial fixtures fail and reference solutions
pass under external acceptance checks.

## 2026-10-08 — Normalized turn-trace instrumentation smoke

The shared runtime now accepts an optional trace collector; real-task reports
always supply one. `trace_format: normalized-turn-v1` adds `turn_trace` without
removing the existing tool trace or altering the application response contract.
It captures each normalized provider response before finish-reason and budget
handling, including blank replies and tool-call batches; exact recovery and tool
messages carry iteration and call IDs. Request boundaries record the requested
output allowance and estimated prompt tokens alongside each response's reported
usage. Initial messages are copied once, so subsequent request transcripts can
be reconstructed without duplicating full history every iteration. Provider
failures retain only exception class. No raw provider payload/headers or exception
text is stored. See `real_tasks/README.md` for format and reconstruction rules.

This changes evidence collection only, with no new prompts, schemas, permission,
or recovery allowance. It is not another five-task quality comparison. Four
**real-tool instrumentation smoke attempts** exercised `divide_zero` and
`parse_flags` in both prompt modes once, using disposable fixture copies and
immutable external acceptance. Initial fixtures failed and reference solutions
passed before inference. This is distinct from live-model simulated-tool smoke
and scripted contracts.

| Field | Value |
|---|---|
| Implementation revision | `7e2a5f3` |
| Report revision status | Dirty: only `evals/real_tasks/README.md` was edited during inference; implementation/tests were committed and unchanged |
| Local model | `gemma4:12b` Q4_K_M, ID `4eb23ef187e2`, Ollama 0.33.2, Apple M1 Pro, 16 GB |
| Settings | Thinking off, server confirmed context 4,096, temperature 0; whole-file tools and configured `python3 -m pytest test_visible.py` |
| Budgets | Eight iterations, 24 tool attempts, one completion retry, 600 s wall dispatch deadline, 300 s request timeout; token gates, required planning, and required verification disabled |
| Comparability audit | All four runtime configurations, tool schemas, prompt versions, and fixture hashes match their cells in the #39 report |
| Raw report | `results/2026-10-08-gemma4-12b-turn-traces-smoke.json` |

| Task/mode | External acceptance | Turn status | Verification | Requests | Blank replies without calls | Injected recovery | Latency |
|---|---|---|---|---|---|---|---|
| divide_zero / harness | passed | budget exhausted (iteration limit) | passed | 8 | 0 | 0 | 56 s |
| divide_zero / minimal | passed | completed | passed | 8 | 1 | 1 | 44 s |
| parse_flags / harness | passed | completed | passed | 7 | 1 | 1 | 66 s |
| parse_flags / minimal | failed | incomplete | not run | 4 | 2 | 1 | 35 s |

All 27 requests have matching normalized responses; all 21 requested tool IDs
match the 21 ordered tool-result messages and legacy tool records, including the
last harness division batch before the iteration limit. Four blank no-call
responses are now directly observable. Minimal division and harness flags each
have one injected recovery followed by a non-empty completed reply; minimal flags
has a second blank reply after its single recovery and returns the explicit
incomplete runtime fallback. Three final artifacts passed acceptance, including
one turn with an incomplete status. These counts describe this smoke run, not
reconstructed counts for historical reports or evidence of a quality gain.

Estimates remain labeled estimates and differ from reported usage (for example,
harness division's first request estimated 2,677 prompt tokens, reported 1,614).
Tracing adds copying/serialization overhead. It retains full normalized content
in memory under existing turn limits, without a separate trace byte budget or
redaction of model-visible content. Reports still checkpoint after each attempt;
a process crash can lose the current attempt. Hidden reasoning, transport bytes,
and durable application sessions are not captured. The evaluation-owned model
was unloaded and server stopped afterward; no paid calls were made.

Reproduce the instrumentation smoke with the prior report's non-secret settings:

```sh
uv run python - <<'PY'
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from evals.real_run import load_tasks, validate_task, run_task, write_report
from harness.config import Settings
from harness.providers import build_configured_provider

config = json.loads(Path(
    "evals/results/2026-10-07-gemma4-12b-argv-feedback-ctx4k.json"
).read_text())["results"][0]["runtime_config"]
config.pop("tool_specs")
settings = Settings(_env_file=None, ollama_model="gemma4:12b", **config)

async def main():
    tasks = [t for t in load_tasks() if t.id in ("divide_zero", "parse_flags")]
    for task in tasks:
        await validate_task(task)
    provider = build_configured_provider("ollama", settings)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    results = []
    try:
        for task in tasks:
            for mode in ("harness", "minimal"):
                results.append(await run_task(
                    task, provider=provider, settings=settings, mode=mode
                ))
                write_report(Path("/tmp/agent-harness-turn-traces-smoke.json"), started, results)
    finally:
        await provider.aclose()

asyncio.run(main())
PY
```

Validation: 82 focused tests and 516 offline tests passed (five live deselected);
Ruff and mypy passed. Contracts reconstruct provider requests, verify deep-copy
isolation and exclusion of raw payloads/exception text, retain rejected final
batches and truncated calls, cover request timeout/cancellation, and compare
traced versus untraced prompts/output allowances/outcomes. Both scripted matrices
passed 90/90 escalation decisions with their `scripted-contract` labels and metric
limitations inspected. These tests establish mechanics, not model capability.
Further bounded/failed-check recovery, context management, and personal-use
persistence/cancellation/reviewable diffs remain open.

## 2026-10-08 — `gemma4:12b`, required verification, context 4,096

This comparison enables the existing `require_verification_before_finish` setting
while holding the preceding string-argv comparison's other recorded settings,
tool schemas, prompt versions, and fixture hashes fixed. A final reply after a
successful file-tool edit must have a passing exact configured check after the
latest edit. Otherwise the shared loop can inject one bounded corrective message;
blank-answer and verification recovery consume the same completion-retry counter.
The iteration/tool/token/wall gates and provider finish-reason precedence remain.
Read-only turns do not acquire a verification requirement.

No implementation, evaluation-runner behavior, or default changed in this PR.
The experiment exercises the shared application path already shipped, using
#40's normalized traces to distinguish recovery from merely observing valid
checks. It does not add a failed-check repair strategy or increase a budget.

| Field | Value |
|---|---|
| Harness revision | `a711d76` (clean, #40 merged) |
| Model/server/hardware | Local `gemma4:12b` Q4_K_M, ID `4eb23ef187e2`, Ollama 0.33.2, Apple M1 Pro, 16 GB |
| Settings | Thinking off; server confirmed context 4,096; temperature 0; whole-file tools; exact `python3 -m pytest test_visible.py` project check |
| Changed setting | `require_verification_before_finish`: false → true |
| Budgets | Eight iterations, 24 tool attempts, one shared completion retry, 600 s wall dispatch deadline, 300 s request timeout; token gates and required planning disabled |
| Attempts | Five tasks × two modes × three repeats = 30, all included; alternating mode order |
| Historical comparator | `results/2026-10-07-gemma4-12b-argv-feedback-ctx4k.json` (#39) |
| Raw report | `results/2026-10-08-gemma4-12b-required-verification-ctx4k.json` |

Reproduce using the prior report's non-secret settings and a unique report path:

```sh
uv run python - <<'PY'
import json
from pathlib import Path
from evals import real_run
from harness.config import Settings

config = json.loads(Path(
    "evals/results/2026-10-07-gemma4-12b-argv-feedback-ctx4k.json"
).read_text())["results"][0]["runtime_config"]
config.pop("tool_specs")
config["require_verification_before_finish"] = True
settings = Settings(_env_file=None, ollama_model="gemma4:12b", **config)
real_run.get_settings = lambda: settings
real_run.main([
    "--provider", "ollama", "--mode", "both", "--repeats", "3",
    "--report", "/tmp/agent-harness-required-verification.json",
])
PY
```

| Task | harness (optional → required verification) | minimal (optional → required verification) |
|---|---|---|
| decimal_total | 0/3 → 0/3 | 0/3 → 0/3 |
| divide_zero | 3/3 → 3/3 | 3/3 → 3/3 |
| parse_flags | 3/3 → 3/3 | 0/3 → 0/3 |
| slugify | 0/3 → 0/3 | 0/3 → 0/3 |
| stable_dedupe | 3/3 → 3/3 | 2/3 → 2/3 |
| **total** | **9/15 → 9/15** | **5/15 → 5/15** |

| Measure | harness | minimal |
|---|---|---|
| Runtime statuses | 6 completed, 9 budget exhausted | 4 completed, 4 budget exhausted, 4 blocked, 3 incomplete |
| Terminations | 6 accepted, 9 iteration limit | 4 accepted, 4 iteration limit, 4 blocked, 3 incomplete |
| Completed status with failed acceptance | 0 (unchanged) | 0 (unchanged) |
| Attempts with a passing configured check | 12/15 (previously 13/15) | 7/15 (unchanged) |
| Final verification statuses | 9 passed, 3 stale, 3 failed | 5 passed, 2 stale, 8 not run |
| Accepted artifacts with incomplete turns | 3 (previously 2) | 1 (previously 4) |
| Median latency | 62 s (previously 59 s) | 46 s (previously 44 s) |
| Median tool-call records | 8 (previously 7) | 7 (unchanged) |
| Median reported tokens | 16,870 (previously 16,859) | 14,479 (previously 14,957) |

### Recovery evidence and limitations

**One verification-specific retry was observed.** Harness flags repeat 2 emitted
`grep_repo`, `read_file`, `emit_plan`, and a correct `write_file`, then an empty
no-call reply at zero-based iteration 4 without a check. The runtime injected
one message combining the non-empty-answer requirement and the exact configured
check. At iteration 5 the model ran that check successfully; at iteration 6 it
returned a non-empty completed answer. Independent acceptance passed. The artifact
was already correct before feedback, and blank recovery would have occurred even
with verification optional. This observed sequence does not prove the extra check
instruction was necessary or improved patch correctness.

**The gate had little opportunity to act.** Replaying the ordered tool records
at every no-call response found only that one post-edit response lacking a current
passing check. No non-empty final reply was rejected for missing verification.
The other 29 attempts received no verification-specific guidance. Thirteen turns
reached the iteration limit, four blocked on repeated commands, and three minimal
flags turns exhausted blank recovery without editing. A completion-time check gate
cannot repair a tool loop that never offers a final reply.

**Full traces make the recovery counts observable.** All 218 requests have
normalized responses, with reported prompt/output usage; all 195 requested tool
IDs match the 195 ordered tool-result messages and legacy records. Responses have
195 `tool_use` and 23 `stop` finish reasons, with no provider truncation/failure in
this run. Thirteen blank no-call replies produced ten injected recovery messages:
nine blank-only messages (minimal division, decimal, and flags, three each), and
one combined message for harness flags. Four recovery-bearing turns ended with
non-empty completed replies (three minimal division and harness flags repeat 2);
three minimal decimal turns continued with tools then hit the iteration limit;
three minimal flags turns returned blank again and ended incomplete. Historical
#39 reports lack full responses, so their exact recovery counts are not comparable.

**Correctness failures remain.** All six decimal artifacts still fail independent
empty-input acceptance. Five final verification states are stale and one passed;
all six hit the iteration limit. Three harness slug artifacts still fail
acceptance and their visible check; all stop at the iteration limit. All three
minimal slug turns and minimal dedupe repeat 2 block before editing. Minimal
flags leaves fixtures unchanged. Minimal dedupe repeat 1 has an accepted artifact
but no check or completed reply; repeat 3 checks and finishes cleanly. No blank
answer received completed status. Zero `false_completion` records reflect terminal
statuses, not proof of correctness: 20/30 turns are incomplete and 16 artifacts
still fail acceptance.

Acceptance is unchanged in every task/mode cell. Completion counts moved from
7/15 to 6/15 in harness and 1/15 to 4/15 in minimal, without establishing a
coding-quality or causal completion gain. All three minimal division turns
consume blank recovery **before editing**, finish with a check, and receive no
verification-specific feedback; their better terminal statuses therefore cannot
be attributed to the changed requirement. More broadly, the historical comparison
is unpaired, repeated decisions vary at temperature 0, disposable paths change,
and #40 adds trace collection between reports. Its copying/serialization overhead
is not isolated. The full offline gate also ran during part of the final decimal
pair, so latency differences are descriptive rather than an isolated measurement.

Keep required verification available as the existing explicit check-enforcement
option; this run does not justify making it the default, increasing budgets, or
claiming a reliable coding-quality gain. Earlier bounded recovery remains a
candidate, but needs evidence about where a retry can still fit and must distinguish
initial failing fixtures from newly introduced failures. Failed-check recovery,
context/model profiles, and personal-use persistence, cancellation, resume, and
reviewable diffs remain open. No additional model, repository, or paid call was
used. The evaluation-owned model was unloaded and server stopped afterward.

Validation: the runner checked initial fixtures fail and reference solutions pass
before inference. All 30 settings/schemas/prompt versions/fixture hashes were
audited against #39 with exactly the one behavioral-setting difference. The raw
report was copied byte-for-byte. Eighty-two focused regression tests and 516
offline tests passed (five live deselected), Ruff and mypy passed. Both scripted
scenario matrices passed 90/90 escalation decisions using unique temporary report
paths; their `scripted-contract` labels and metric limitations were inspected.
These checks validate harness mechanics, not model coding quality.

## 2026-10-08 — Early string-argv recovery trial (not retained)

**Decision:** retire the experimental runtime option. Harness acceptance moved
from 9/15 to 10/15 and minimal from 5/15 to 4/15, leaving aggregate acceptance
14/30. Two minimal slug artifacts received completed status despite failing
independent acceptance, versus zero previously. The trial often corrected tool
arguments but did not establish a reliable coding-quality gain. The final branch
restores the prior runtime, defaults, tests, and evaluator behavior. The frozen
trial implementation and raw report remain available for inspection and replay;
`recover_string_argv` is **not a supported setting on the final branch**.

The observed obstacle was early malformed command input: #41's normalized traces
show strings first failing as early as iteration 2 and repeating until a block
or iteration limit. The trial introduced a default-off shared setting that, after
a string-valued `run_command.argv` validation error, injected a concise user
message with an actual argument-object example and the exact configured check.
It did not parse strings, rewrite arguments, dispatch invalid input, add tools,
change system prompts, or increase budgets. The model had to correct its own call.

Feedback followed all tool results in a batch; a later array call in that batch
superseded the format reminder. No prompt was added after a dispatch stop or on
the last iteration. Early, blank-answer, and missing-verification recovery shared
one counter. A consumed early allowance could therefore leave a later final reply
without recovery. Subsequent provider/tool actions remained subject to the existing
iteration, tool, wall, and token gates. This trial targeted string arguments, not
failed checks, incorrect executable tokens inside arrays, or semantic patch repair.

| Field | Value |
|---|---|
| Frozen trial revision | `c2d6626b504371a2feccbe67e4c813679dbf0567` (clean) |
| Retirement revision | `e67107f` (reverts the trial's code/tests) |
| Local model | `gemma4:12b` Q4_K_M, ID `4eb23ef187e2`, Ollama 0.33.2, Apple M1 Pro, 16 GB |
| Settings | Thinking off; server confirmed context 4,096; temperature 0; whole-file tools; exact `python3 -m pytest test_visible.py`; required verification enabled |
| Trial setting | `recover_string_argv=true`, added only at the frozen revision |
| Budgets | Eight iterations, 24 tool attempts, one shared recovery allowance, 600 s wall dispatch deadline, 300 s request timeout; token gates and required planning disabled |
| Attempts | Five tasks × two modes × three repeats = 30; all included, alternating mode order |
| Comparator | `results/2026-10-08-gemma4-12b-required-verification-ctx4k.json` (#41) |
| Raw report | `results/2026-10-08-gemma4-12b-early-argv-ctx4k.json` |

The filename uses the local experiment date, October 8. The report's UTC start
is October 9. All settings, schemas, prompt versions, and fixture hashes were
audited against #41, with only the added early-recovery setting enabled. Fixtures
and references were validated before inference; authoritative acceptance checks
remained outside fresh disposable copies. The raw report was copied byte-for-byte.

| Task | harness (#41 → early recovery) | minimal (#41 → early recovery) |
|---|---|---|
| decimal_total | 0/3 → 0/3 | 0/3 → 0/3 |
| divide_zero | 3/3 → 3/3 | 3/3 → 3/3 |
| parse_flags | 3/3 → 3/3 | 0/3 → 0/3 |
| slugify | 0/3 → 1/3 | 0/3 → 0/3 |
| stable_dedupe | 3/3 → 3/3 | 2/3 → 1/3 |
| **total** | **9/15 → 10/15** | **5/15 → 4/15** |

| Measure | harness | minimal |
|---|---|---|
| Runtime statuses | 6 completed, 9 budget exhausted | 6 completed, 6 budget exhausted, 3 incomplete |
| Terminations | 6 accepted, 9 iteration limit | 4 accepted, 2 acceptance failed, 6 iteration limit, 3 incomplete |
| Completed status with failed acceptance | 0 (unchanged) | 2 (previously 0) |
| Attempts with a passing configured check | 12/15 (unchanged) | 9/15 (previously 7/15) |
| Final verification statuses | 11 passed, 2 stale, 1 failed, 1 not run | 7 passed, 3 stale, 1 failed, 4 not run |
| Accepted artifacts with incomplete turns | 4 (previously 3) | 0 (previously 1) |
| Executed commands | 21 (unchanged) | 14 (previously 7) |
| Median latency | 62 s (unchanged rounded) | 51 s (previously 46 s) |
| Median tool records | 8 (unchanged) | 7 (unchanged) |
| Median reported tokens | 16,880 (previously 16,870) | 14,856 (previously 14,479) |

### Observed recovery and correctness

Eleven early messages were injected in eleven distinct attempts. Ten later
submitted a valid argv array: nine did so on the immediately following model
iteration; minimal dedupe repeat 1 first repeated a quoted array, then sent a
valid array naming a nonexistent `test_dedupe.py` before finding the configured
file. Minimal dedupe repeat 3 never corrected its strings. A later array is not
equivalent to an executed relevant check, a passing check, or a correct artifact.

String-argv validation errors declined from 19 to 15 (harness 3→4, minimal 16→11).
Repeated-call blocks declined from four to zero, but their removal did not yield
accepted minimal slug artifacts. Two malformed executable-token arrays in harness
slug repeat 1 remained rejected by the allowlist and did not trigger this narrow
recovery. The trial recorded 222 requests and responses, 196 requested call IDs
matching ordered tool-result messages/legacy records, and 22 recovery messages:
11 early, eight blank-only, and three combining blank-answer and verification.
There were 14 blank no-call responses. The shared allowance was never increased.

| Early-feedback attempt | Later argv / terminal outcome |
|---|---|
| minimal division 1 | Immediate valid configured check; accepted artifact and completed turn |
| harness dedupe 1 | Immediate valid initial failing check, then edit and passing check on last iteration; accepted artifact, incomplete turn |
| harness decimal 1/3 | Immediate valid configured check, later edit without rerun; failed acceptance, stale verification, iteration limit |
| harness slug 2 | Immediate valid passing visible check, later rewrite and another pass; accepted artifact and completed turn |
| minimal slug 1/2 | Immediate valid initially failing check, edit and passing visible check; completed turns, failed acceptance |
| minimal slug 3 | Immediate valid initially failing check, edit on last iteration; failed acceptance and stale verification |
| minimal dedupe 1 | Initially repeats a string, then guessed filename, then configured check; unchanged fixture, failed acceptance, iteration limit |
| minimal dedupe 2 | Immediate valid configured check; accepted artifact and completed turn |
| minimal dedupe 3 | Strings persist despite feedback; unchanged fixture, failed acceptance, iteration limit |

Independent replay of successful file-tool writes **before** the early message
confirmed minimal division repeat 1 and minimal dedupe repeat 2 already passed
acceptance. Their successful call correction does not explain creation of a
working patch. Harness slug repeat 2's pre-feedback artifact failed underscore
acceptance even though the subsequent visible check passed; a later rewrite
preserved underscores through separator conversion and passed independent
acceptance. The trace establishes temporal order, not that the feedback caused
that semantic repair. No hidden reasoning or counterfactual model run is available.

All six decimal artifacts still fail empty-input acceptance; five final check
states are stale and one passed, and all six hit the iteration limit. All three
minimal slug artifacts still discard underscores; repeats 1/2 now have passing
configured checks and non-empty completed answers but fail independent acceptance.
Harness slug repeats 1/3 also fail underscore acceptance. Minimal dedupe repeats
1/3 remain unchanged; repeat 2 was correct before feedback. Minimal flags remains
unchanged and exhausts blank recovery in all three attempts. The earlier prompt
helps some argument formatting without establishing patch correctness.

This historical comparison is unpaired, small, and confined to five fixtures.
Decisions varied before changed feedback despite temperature 0, with fresh
workspace paths; counts are observations, not a causal gain or regression claim.
Correcting calls can expose semantic failures that blocked runs never reached.
More passing visible checks and fewer repeated-call blocks did not improve total
acceptance, and completed failed artifacts increased. Eighteen turns still ended
incomplete, with four accepted artifacts among them; 16 artifacts still failed
acceptance. Latency differences are descriptive. No premium supervisor, extra
model, paid call, or additional repository was used. The evaluation-owned model
was unloaded and server stopped after inference.

The roadmap retains demonstrated improvements or changes required for correctness.
This duplicate recovery cue does not meet that retention bar; it is removed rather
than becoming an extra supported runtime option. Existing accurate tool validation
feedback, bounded completion recovery, required verification, and normalized
traces remain. Failed-check recovery, model/context profiles, and personal-use
persistence, cancellation, resume, and reviewable diffs remain open.

### Replay and validation

Replay **only at the frozen implementation**, in an isolated worktree so existing
user edits are preserved. The final branch does not implement the trial option:

```sh
git worktree add --detach /tmp/agent-harness-early-argv-replay c2d6626b504371a2feccbe67e4c813679dbf0567
cd /tmp/agent-harness-early-argv-replay
uv run python - <<'PY'
import json
from pathlib import Path
from evals import real_run
from harness.config import Settings

config = json.loads(Path(
    "evals/results/2026-10-08-gemma4-12b-required-verification-ctx4k.json"
).read_text())["results"][0]["runtime_config"]
config.pop("tool_specs")
config["recover_string_argv"] = True
settings = Settings(_env_file=None, ollama_model="gemma4:12b", **config)
assert settings.recover_string_argv is True
real_run.get_settings = lambda: settings
real_run.main([
    "--provider", "ollama", "--mode", "both", "--repeats", "3",
    "--report", "/tmp/agent-harness-early-argv-replay.json",
])
PY
```

Use an unused worktree/report path and the configured local model. The replay
requires an Ollama server; only stop processes owned by that evaluation. Historical
report settings must not be loaded into the final branch to silently simulate
this experiment: unknown settings are ignored by `Settings` there. The assertion
above makes an unsupported replay fail explicitly.

The frozen trial passed 95 focused and 529 offline tests (five live deselected),
Ruff, and mypy. Its regressions cover rejected strings without process dispatch,
model-submitted array correction, disabled/zero/exhausted recovery, last-iteration
and tool/token gates, provider truncation, batch supersession, and sharing the
allowance with later blank/verification recovery. Both scripted matrices passed
90/90 with `scripted-contract` labels and metric limitations inspected. After
retirement, the final runtime passed 516 offline tests (five live deselected),
Ruff, mypy, and both 90/90 scripted matrices, with their evidence labels inspected.
These validate mechanics, not model quality. No trial tests or behavior
are shipped in the final diff.


## 2026-10-12 — Duodoro duration formatting, `gemma4:12b`

One authorized real-project library bugfix pinned at Duodoro
`5324cabba4ad3c8ad2e1cd1480218cf087d9b741` was attempted four times using shared
runtime/real tools in disposable workspaces. Preparation and external acceptance
are documented in `projects/duodoro/README.md`. The initial artifact fails and the
reference passes visible and independent checks. Criterion hashes stayed unchanged;
source checkout and synthetic pre-existing work were preserved. No manual patch rescue.

| Field | Value |
|---|---|
| Harness revision | `e7181b9111a437b2471ca24759dfb31f2e36178f`, clean |
| Model | `gemma4:12b`, Ollama 0.33.2; digest `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c` |
| Hardware | Apple M1 Pro, 16 GB unified memory |
| Generation | temperature 0, thinking off, configured server context 16,384 |
| Budgets | 8 iterations, 24 tool calls, 600 s dispatch deadline, 180 s request timeout, 8,192 output tokens, 16,384 estimated context, one completion retry |
| Policies | Full toolset; required verification, no required plan; exact check `python -m pytest -q test_visible.py` |
| Prompts | harness `coding-v2`, minimal `minimal-v1`; harness still contains legacy support instructions |
| Environment | Managed Node 24.21.0 native TypeScript adapter; no npm install, Next/Vitest or application integration suites |

| Total-token budget | Order | Mode | Accepted | Terminal status | Prompt/output tokens | Configured-turn seconds |
|---|---|---|---|---|---|---|
| 32,000 | first | harness | 0/1 | incomplete, empty final answer | 4,308 / 18 | 31.0 |
| 32,000 | second | minimal | 0/1 | budget exhausted | 24,775 / 95 | 95.5 |
| 64,000 | first | minimal | 0/1 | incomplete, empty final answer | 3,878 / 18 | 29.3 |
| 64,000 | second | harness | 0/1 | incomplete, empty final answer | 4,310 / 18 | 22.1 |

Raw reports: `results/2026-10-12-duodoro-duration-gemma4-12b-32k.json` and
`results/2026-10-12-duodoro-duration-gemma4-12b-64k.json`. All artifacts stayed
unchanged and failed behavioral acceptance; scope preservation passed. No check ran.
The 32K minimal trace has four tool records: three executed reads/tree calls, then
one proposed read blocked before execution because the conservative next-request
estimate could not fit the remaining total budget. Other attempts exhausted blank
recovery before tools. None were reported completed or check-passed.

The exploratory 64K variant changed only total-token budget and reversed mode order.
One repeat per cell on one task, with unmeasured load/cache conditions, establishes
no prompt, budget or model-quality gain. Recorded turn duration includes configured setup/final workspace review and excludes
external acceptance, preparation, report writing and transport. Cleanup/final review can
exceed the dispatch deadline. Full application regression coverage,
Node 22 compatibility, held-out tasks and the ten-task release target remain open.

The temporary 32K wrapper initially mislabeled no-edit failures as out-of-scope
changes. Postprocessing restored original runtime termination labels and summary;
the report notes the correction. No artifacts were repaired or calls repeated.

Separately, `results/2026-10-12-duodoro-duration-response-probe.json` is one
response-only live-model simulated-tool smoke probe: no tools executed, 256-token
output cap. Its upstream Ollama payload had empty content, no thinking field or tool
calls, and stop (2,135 prompt / 11 output tokens). Cause remains unresolved; this
is not coding-quality evidence. The provider did not discard populated content in
that probe. Evaluation-owned Ollama processes were stopped; user services preserved.


## 2026-10-12 — Duodoro duration formatting, OpenCode Go free preview

Jorge authorized a cheaper model through his existing OpenCode account within limits.
The current official [Go documentation](https://opencode.ai/v2/docs/console/go/) and
installed CLI metadata listed `longcat-2.5-preview-free` at zero input/output/cache
price on 2026-10-12. Two attempts used the existing credential privately, without
OpenCode's agent, a premium supervisor, SDK retries, paid fallback or billing changes.
Actual account billing was not independently observed. Preview availability/pricing
can change; verify it again before replay, and stop on provider/limit errors.

Raw report: `results/2026-10-12-duodoro-duration-longcat-free-64k.json`.
Harness revision `007c53a84becd0fac390caf3d6c8ef9397f5b3ff`, clean. Endpoint
`https://opencode.ai/inference/go/v1` came from current installed CLI public metadata.
The OpenAI-compatible adapter sent its own `agent-harness-real-eval/0.1` User-Agent
and a fresh UUID in `x-opencode-session` per attempt, stable for that conversation.
The adapter's explicit header seam is not automatic session routing for the API.
SDK retries were zero and request timeout 180 seconds; comparison stops at the first
provider error. None occurred. Credentials/header session values are absent from reports.

The pinned task, external acceptance, full tools, prompt modes, temperature 0,
16,384 estimated context, 64,000 total-token estimate gate, 8,192 output-token cap,
8 iterations, 24 calls, 600-second dispatch deadline and verification policy match
the preceding Gemma 64K variant. No backend-specific thinking option was sent;
hosted reasoning/context defaults and the provider's served model revision are unknown.
Managed Node library coverage and source-preservation limits remain the same.

| Mode (execution order) | Acceptance | Runtime status / checks | Prompt/output tokens | Configured-turn seconds |
|---|---|---|---|---|
| minimal (first) | 1/1 | budget_exhausted / not_run | 53,548 / 2,050 | 54.4 |
| harness (second) | 0/1 | budget_exhausted / not_run | 47,638 / 883 | 34.8 |

Minimal inspected the instructions, target, tests and adapter, received one rejected
`node --version` command, then made two exact edits only to `format.ts`. External
acceptance passed behavior and scope preservation. The total-token gate blocked the
next proposed read before execution, so no configured check or final success claim
occurred. Its ten records include nine dispatched tools (including the rejected
command) and one budget-blocked call. This is an independently accepted artifact,
not a harness-verified or completed turn; no manual repair was applied.

Harness performed 13 dispatched inspection/error calls and no edits. An estimated
16,935-token next prompt could not fit its 16,384 context with the required output
reserve. Errors included absent Git metadata in the exported fixture, rejected
`node --version`, and excluded `client/node_modules`. External behavior failed;
scope preservation passed. Criterion hashes and the user's Duodoro checkout stayed
unchanged in both attempts. Provider clients were closed afterward.

One accepted minimal artifact versus zero harness artifacts is a useful observation
about this task, not a ranking or measured improvement. One repeat per model/mode,
one task, different backends/defaults and sequential unmeasured cache/load conditions
cannot satisfy the roadmap's repeated two-model, held-out or ten-real-task criteria.
Observed context pressure supports investigating bounded context/prompt profiles;
it does not justify silently enlarging limits or adding a supervisor.

To reproduce the transport, prepare/validate the pinned task as described in its
README, reconstruct `Settings(_env_file=None, **runtime_config)` after removing
`tool_specs`, set the report's compatible URL/model, and create a separate provider
per mode with `create_chat_provider("openai", provider_name="openai_compatible",
api_key=<privately loaded Go key>, model="longcat-2.5-preview-free", base_url=<URL>,
timeout_seconds=180, max_retries=0, default_headers={"User-Agent":
"agent-harness-real-eval/0.1", "x-opencode-session": str(uuid4())})`.
Use the shared `run_task`, close each provider, stop on `error:` termination, require
unchanged criterion hashes, and reject added/deleted or out-of-scope changed files.
Retain all results via `write_report` at a unique external path. Do not print/persist
the key or reuse one header session across separate conversations.


## 2026-10-12 — Duodoro timer normalization, OpenCode Go free preview

Raw report: `results/2026-10-12-duodoro-timer-longcat-free-64k.json`.
Two `longcat-2.5-preview-free` attempts used the same transport, declared-free
pricing, bounded settings and minimal-then-harness order as the duration Go trial.
Harness revision `dec1009bf5bc9d50a125f21c481806096e7f98b3`, clean. The second frozen
Duodoro task/criterion passed initial-fail/reference-pass validation before inference.
The user's source checkout stayed unchanged and external criterion hashes matched
before/after each attempt. No manual rescue, paid fallback, SDK retry or provider error.
Each conversation had its own stable session header and its provider was closed.

| Mode | Accepted | Status / checks | Prompt/output tokens | Configured-turn seconds | Dispatched tools |
|---|---|---|---|---|---|
| minimal | 0/1 | budget_exhausted / not_run | 31,951 / 1,559 | 35.3 | 7 |
| harness | 0/1 | incomplete (truncated) / not_run | 30,534 / 2,807 | 56.7 | 5 |

Both artifacts remained unchanged and failed behavioral acceptance, while exact
scope preservation passed. Minimal read instructions, source/tests and companion
names, received an explicitly incomplete broad-search error, narrowed its inspection,
then stopped before a model request estimated at 16,690 tokens in a 16,384 context.
Harness inspected directories/instructions/tests/source, then received finish-reason
`length` with empty visible final content and no edit. The raw normalized response
records the output allowance and usage; this does not establish hidden reasoning,
served-model limits, or a backend bug. No check ran and neither turn claimed completion.

Across these two Duodoro tasks, free Go yielded one independently accepted artifact
in four attempts (minimal 1/2, harness 0/2), with no completed/checked turns. These
are small observations, not repeated comparisons, a held-out score, a ranking, or a
quality improvement. Timer has not been run with Gemma. Context pressure/truncation,
inspection choices and check completion remain concrete obstacles; preserving limits
and failure evidence is preferable to silently increasing budgets. Native Node library
criteria still omit full project integration; account billing remains unobserved.


## 2026-10-13 — Paired compact prompt trial, free Go, two Duodoro tasks

Raw report: `results/2026-10-13-longcat-free-compact-duodoro.json`.
Frozen implementation `8becd55a0cfd5a7497dc3ad2494d44f26c22d7e1`, clean.
Same `longcat-2.5-preview-free`, pinned source/criteria, full tools, temperature 0,
16,384 context estimate, 64,000 total tokens, 8,192 output tokens, 8 iterations,
24 calls, 600-second dispatch deadline, 180-second requests and one completion retry.
Verification required, plan not required. Twelve attempts: two tasks × two prompts
× three repeats; alternating pair order is recorded. Every conversation had fresh
isolated workspace/facts and a stable fresh Go session header. SDK retries zero;
stop on provider error; none occurred. Declared-free pricing rechecked October 13;
actual account billing, served model revision and load/cache effects remain unobserved.
No premium supervisor or manual patch repair. Provider clients closed after attempts.

Only the prompt contract changed: legacy coding-v2 (1,044 UTF-8 bytes, 158 words)
versus compact coding-v3-compact-v1 (640 bytes, 94 words), removing support/memory
instructions and emphasizing bounded inspection, preservation, check argv and honest
reporting. These are byte/word counts, not tokenizer counts. Tools, schemas, budget/
recovery policy, injected check and task requests remained unchanged. Both profiles
used harness mode; the generic report summary aggregates them, while
`summary_by_profile`, per-result config/version and `attempt_order` distinguish them.
Earlier minimal-prompt trials are separate cohorts, not paired controls here.

| Task | Legacy accepted | Compact accepted | Compact completed / latest check passed |
|---|---|---|---|
| duration formatting | 1/3 | 3/3 | 1/3 completed; 3/3 checks passed |
| timer normalization | 0/3 | 1/3 | 0/3 completed; 1/3 checks passed |
| Total | 1/6 | 4/6 | 1/6 completed; 4/6 checks passed |

Legacy's accepted duration repeat 1 ran an initially failing check, then edited and
exhausted total tokens without rerunning; its check was stale. The other legacy
artifacts stayed unchanged: three context-budget stops and two length truncations
(one with a failed check). No legacy turn completed or had a latest passed check.
Compact duration repeat 2 completed, passed its configured check and external
acceptance. Its other two duration successes and timer repeat 1 passed both checks
but exhausted total tokens before completion; those are accepted partial artifacts,
not completed turns. Timer compact repeats 2/3 inspected without editing and hit
estimated context. No failed artifact was labeled completed. All external criteria
stayed intact and all scope assertions passed; evaluation writes stayed in disposable
copies. No Duodoro working-tree commits or pushes were made by this trial.

Observed median configured-turn time was 44.3 s legacy / 60.0 s compact; median
reported prompt+output tokens 42,128.5 / 50,496. More accepted work here also used
more time/tokens. These timings include configured setup/final workspace review,
excluding external acceptance, preparation, report writing and transport. They do
not establish causal efficiency gains or a hard return deadline.

The compact profile is retained **opt-in**, with legacy still default, on the strength
of these bounded artifact/check observations and a smaller coherent coding contract.
This is two tuning tasks, not a held-out benchmark, general model ranking, statistical
proof or completed Milestone 4/5/6. Temperature-zero repeats varied; sequential
backend/cache effects and fresh workspace paths are uncontrolled. Gemma has not been
compared with compact. Most accepted artifacts still failed to finish within the
budget; context management and broader repeated same-model tests remain open.

### Runtime-label correction and replay

Earlier Duodoro reports labeled Node 26.0.0 from a shell probe. Failed check outputs
show v24.21.0, and a direct `run_process(["node","--version"])` confirms the managed
command environment selects it while the shell reports 26.0.0. Raw metadata now
retains the original label and adds correction/provenance notes; no inference,
artifact or acceptance result was changed. Per-invocation versions were not separately
captured, so do not treat this as runtime attestation. The adapters omit Next/Vitest,
full server/browser/integration suites and TypeScript typechecking; Node 22 compatibility
is unverified. Future evaluation setup should probe the actual managed environment.

Replay at the frozen implementation in an isolated worktree, prepare/validate both
pinned tasks, reconstruct the published runtime config after removing `tool_specs`,
set the public compatible endpoint/model, and explicitly assert
`settings.coding_prompt_profile` is the requested legacy/compact value. Use shared
`run_task(mode="harness", attempt=repeat)` with the report's alternating order and
three repeats, privately loaded Go credential, one provider per conversation, own
User-Agent/stable UUID, and zero SDK retries. Verify free pricing first, stop on
provider errors, verify criterion hashes and veto out-of-scope file changes. Write
all results to a unique external path; group by profile rather than the common mode.
Frozen trial validation: 103 focused, 870 offline tests (five live deselected), Ruff,
mypy, formatting and both scripted 90/90 matrices with evidence limits inspected.
Those tests measure mechanics, not model quality.

## 2026-10-13 — Bounded read-context trial stopped and retired

Raw report: `results/2026-10-13-longcat-free-bounded-context-duodoro.json`.
Frozen trial implementation `d411ded636e90257f3483095878354de90ded98e` is available
for isolated replay; **the runtime option and its tests were subsequently removed**.
The trial's deterministic request projection kept the last eight messages, all
AGENTS.md reads, instructions/objectives, edits/diffs, errors and command/check
output. Older native read results above 2,048 content characters kept their first/
last 512 characters, path/hash/range and an explicit omission/reread notice.
Raw transcripts/results stayed complete. Request and review estimates used the
same view; changed prefixes reset only the prompt anchor, retaining observed usage.
This is read elision, not semantic compaction or a complete context strategy.

The planned comparison was two frozen Duodoro tuning tasks × two modes × two
repeats, with alternating pair order. Both modes used compact prompt/full tools,
Go `longcat-2.5-preview-free`, temperature zero, 16,384 estimated context,
64,000 total tokens, 8,192 output tokens, eight iterations, 24 tools, 600-second
configured dispatch deadline, 180-second requests, one completion retry and required
verification. Source/criteria, settings and explicit manual Go transport recipe
matched between arms: own User-Agent, fresh stable conversation UUID, zero SDK
retries. This trial uses that constructor recipe, not the new configured Go mode.
[Go documentation](https://opencode.ai/v2/docs/console/go/) listed free preview
input/output/cache pricing on October 13. Actual billing/backend/cache/load and
served revision remain unobserved; no paid fallback or premium supervisor.

Only three of eight attempts ran, all on duration formatting:

| Attempt | Accepted artifact | Completion / check status | Reduced requests | Configured-turn seconds |
|---|---|---|---|---|
| repeat 1, full | yes | budget exhausted / failed | 0 | 66.5 |
| repeat 1, bounded reads | yes | budget exhausted / failed | 0 | 53.6 |
| repeat 2, bounded reads | no, unchanged | incomplete / not run; ProviderError | 0 | 180.1 |

Both accepted patches passed external acceptance and scope checks without manual
repair. Their requested configured check was **blocked by the total-token gate**,
so failed status is a blocked invocation, not an executed test failure. Neither
completed. The third attempt failed at the first request before tools; normalized
trace records ProviderError. The underlying cause is unobserved; elapsed time near
the configured timeout does not establish its cause. Zero reported tokens on that
attempt mean no response usage observed, not proof that the backend did no work.
The runner stopped the entire comparison at that first provider error without
retry/fallback. No second full repeat or timer attempts were run. Clients closed;
criterion hashes and file/mode scope assertions stayed intact.

The single completed pair did **not activate reduction**. Its token/timing/patch
variation cannot be attributed to the new mode. The incomplete, unequal cohorts
must not be presented as a quality ranking. The roadmap retains only demonstrated
improvements or basic correctness changes: this trial establishes neither a
model-quality gain nor a reason to ship this particular reduction policy. Retire
it, retain the frozen evidence, and leave context management/Milestone 4 open.
No inference was used to repair any evaluated patch. All evaluation writes stayed
in disposable copies; future Duodoro source changes require branches and PRs.

The process imported the frozen implementation before inference. README and a
nonfunctional context comment were edited during the run; the report's resulting
dirty flag is preserved with that provenance, rather than labeled clean. No
functional runtime edits occurred during inference. Managed Node 24.21.0 comes
from earlier managed-process observations, not per-call attestation; library
adapters omit full application/Vitest/TypeScript/Node22 coverage.

Trial mechanics validation: 124 focused and 888 offline tests (five live deselected),
Ruff/mypy, changed-file formatting, and both 90/90 scripted matrices with labels/
metric limits inspected. The initial added fixtures used the wrong response field,
exceeded the native read-line limit and hit the intentional repetition guard; each
was corrected before the final gate. These tests validate request/budget/trace,
profile/review and archive/restart mechanics, not model quality. After retiring the
trial, the restored runtime passed 116 focused and 880 offline tests (five live
deselected), Ruff and mypy; the existing Chroma deprecation warning remained.

For replay, use the frozen revision in an isolated worktree; prepare the published
pinned tasks/criteria, reconstruct per-result settings without `tool_specs`, and
select compact prompt plus full or bounded_reads_v1. Use shared `run_task` in
harness mode, privately configured Go credential and the recorded explicit transport
recipe. Recheck free pricing and stop on provider errors; protect criteria and veto
out-of-scope artifacts. Never reuse the actual Duodoro working tree as a fixture.


## 2026-10-14 — Context-pressure evidence, offline only

New normalized-turn-v2 traces record metadata-only prompt footprints for request
attempts and separate pre-request `budget_stop` records. The shared estimator's
math is unchanged: message tokens include overhead; schema tokens plus message
estimate plus nonnegative reported-anchor adjustment equal the gate estimate.
Rankings cap at ten message payloads by heuristic UTF-8 bytes, with stable index
ordering on ties. They are not provider tokenizer counts or wire sizes. Budget-stop
metadata records observed cumulative usage, configured limits and required output;
it never claims a prevented request dispatched. Old request records parse without
invented footprints and historical raw reports remain unchanged.

An offline real-file regression reaches an actual context gate after a successful
read, preserves user content/full tool results, identifies that read as the largest
payload and prevents a second provider call. A separate real-task report regression
retains an initial prevented request through JSON serialization without inference
or changing external acceptance. Existing traced/untraced parity checks pass.
These use scripted providers and real tools, not model-quality evaluations.

Post-hoc reconstruction of
`results/2026-10-12-duodoro-timer-longcat-free-64k.json` was performed locally without
new inference or modifying its results. Rebuild the normalized transcript from
request initial history, response messages and tool/recovery events; update
TokenBudget from each response's usage and sent message count; apply
`harness.trace.prompt_footprint` with the recorded tool schemas. The minimal
attempt's final-history estimate reproduces the published context stop:

| Component | Reconstructed estimate / heuristic payload bytes |
|---|---|
| Message estimate, including overhead | 13,971 tokens |
| Tool schemas | 2,719 tokens |
| Anchor adjustment | 0 tokens |
| Final prompt estimate | 16,690 tokens, against 16,384 context plus 256 required output |
| Largest message: AGENTS.md read, index 3 | 17,396 payload bytes |
| Search output, index 12 | 6,906 payload bytes |
| Parser source read, index 6 | 6,168 payload bytes |
| Parser tests read, index 4 | 5,030 payload bytes |

File labels come from the existing raw trace; footprints themselves contain no
paths/content/arguments. This is retrospective estimator analysis, not a new
observed request, backend token attestation, latency/cost comparison or quality
result. It explains why source-read elision alone missed substantial pressure;
applicable instructions must remain preserved in any future context experiment.
The historical harness attempt still ended on length truncation; its reconstructed
13,726-token history does not imply another request would succeed.

Validation: 75 focused, 883 offline tests (five live deselected), Ruff/mypy and both
90/90 scripted matrices with labels/metric limits inspected. The new workspace
assertion initially used the wrong status name and was corrected to `tracked`.
No live calls, paid calls, Duodoro source edits or model comparison occurred.
Later post-response/pre-tool budget stops and wall-time conditions are outside the
new stop-record coverage; raw traces/final reasons remain authoritative. Trace
processing adds overhead inside existing deadlines. Context reduction, model
profiles/calibration and Milestones 4–6 remain open.


## 2026-10-14 — Unknown usage after failed evaluation requests

The real-task observation wrapper now marks prompt/completion totals unknown when
an in-flight request raises, times out or is cancelled, including after earlier
successful responses. Known prefix usage remains in raw response traces. Existing
summary logic excludes these incomplete totals from token medians and counts them
as unreported attempts. Runs blocked before any request still retain zero, and
reported zero remains valid. No retry or fallback is introduced.

This fixes the reporting limitation seen in the retired context trial: its failed
first request showed zero tokens because the wrapper had no response. That did not
prove no backend work. Historical raw reports are left intact with their existing
notes; this change affects new evaluations. No model comparison or quality claim.

## 2026-10-17 — Acceptance bootstrap correction (offline)

A disposable unchanged `divide_zero` fixture returned acceptance exit 1 normally,
then exit 0 after adding `pytest.py` containing `raise SystemExit(0)`. The old
`python -m pytest` invocation selected the editable module before the external
acceptance suite ran. No inference was used, and this reproduction does not show
that any historical model attempt used that path.

New evaluations record `acceptance_policy: owned-pytest-v1`: Python `-I -B` starts
an external bootstrap, installed pytest loads before task imports, configuration
is pinned outside the workspace, conftest/plugin autoload and cache are disabled,
and an acceptance path inside the editable workspace is rejected. A passing
model-invoked check can still be rejected by authoritative acceptance. Seven
trusted references remain solvable; shadow/configuration regressions are offline
mechanics evidence only. Old reports keep their original grading limitations and
are not regraded. This is not process isolation: imported implementation code
retains host access. No model-quality improvement, Duodoro edits or paid calls.

## 2026-10-17 — Bounded Gemma development-task envelope

Six live real-tool attempts used three public trusted fixtures at clean harness
revision `8e6e454c93bfa40ef0af6541de01feb71d18da66`: one refactor, one bugfix
with a protected pre-existing failure, and one two-module feature. This is a
small development observation, not a held-out evaluation or a measured gain.
No runtime, prompt or default setting was changed following these attempts.

Evidence:

- [Raw results, normalized traces and bounded final diffs](results/2026-10-17-gemma4-12b-development-envelope.json)
- [Frozen settings, order, file hashes and cleanup](results/2026-10-17-gemma4-12b-development-envelope.metadata.json)
- [Observed loaded-model metadata](results/2026-10-17-gemma4-12b-development-envelope.active-model.json)
- [Frozen inference recipe](results/2026-10-17-gemma4-12b-development-envelope-reproduce.py)
- [Post-hoc saved-diff replay audit](results/2026-10-17-gemma4-12b-development-envelope.replay-audit.json)
- Extended replay diagnostics: [compact](results/2026-10-17-gemma4-12b-development-envelope.replay-harness-failure.txt), [minimal](results/2026-10-17-gemma4-12b-development-envelope.replay-minimal-failure.txt)

The exact model was local `gemma4:12b`, digest
`4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`,
Ollama 0.33.2, on an Apple M1 Pro with 16 GiB RAM. Requests explicitly used
16,384 context and thinking off; the active-model observation also reported
16,384. This observation does not attest every request's backend compliance.
Sampling temperature was 0. The shared application runtime used the full toolset
and compact `coding-v3-compact-v1` versus `minimal-v1`. All six attempts used
`owned-pytest-v1` independent acceptance outside disposable editable workspaces.
Trusted task code still runs with host permissions; copies are not a sandbox.

Each attempt had eight model iterations, 24 tools, a 180-second dispatch wall
budget, 60-second request timeout, 8,192 reported output tokens per turn,
2,048 requested output tokens per request, 64,000 total tokens, 16,384 effective
context and 256 minimum output tokens. Prompt estimation used 3 bytes/token.
One completion recovery and two identical calls were allowed, with four touched
files. Plan/check requirements, root instruction loading and support tools were
off; no project check argv was configured. Thus observed check status uses the
recognized-command heuristic, not a task-specific acceptance command. Settings
came from explicit defaults plus frozen overrides, with dotenv disabled and
isolated unseeded fact stores. Token gates remain estimate-based and the dispatch
wall limit is not a hard end-to-end latency bound.

There was one attempt per task/mode, in the table's order. All planned attempts
are retained, with no model reruns, prompt tuning or manual patch repairs.

| Task | Prompt | Artifact accepted | Completion | Observed check | Seconds | Tools | Prompt / output tokens |
|---|---|---|---|---|---:|---:|---:|
| `extract_label_normalizer` | compact | yes | iteration limit | passed | 90.7 | 7 | 22,810 / 688 |
| `extract_label_normalizer` | minimal | yes | completed | passed | 77.8 | 6 | 20,319 / 630 |
| `positive_total_existing_failure` | minimal | yes | iteration limit | stale | 56.6 | 7 | 20,667 / 311 |
| `positive_total_existing_failure` | compact | yes | iteration limit | passed | 73.3 | 8 | 23,021 / 481 |
| `retry_after_seconds` | compact | no | iteration limit | passed | 118.9 | 8 | 24,301 / 883 |
| `retry_after_seconds` | minimal | no | iteration limit | not run | 89.2 | 7 | 20,850 / 724 |

Both prompts produced accepted final artifacts on **2/3** tasks. Only the minimal
refactor reached a completed answer: completed-and-accepted workflows were
**compact 0/3, minimal 1/3**. Five attempts exhausted iterations; none ended on
provider/request failures, length, context, total-token, output-token or wall
limits. All six had reported usage. The aggregate false-completion count is zero
because the sole completed attempt passed acceptance; this is not a general
correctness guarantee for passing check states. Median latency was 90.7 versus
77.8 seconds, median tools 8 versus 7, and median total tokens 23,502 versus
20,978 (compact versus minimal). One sample per cell, fixed order, a cold first
compact request and shared host conditions prevent a reliable latency or prompt
quality comparison. These counts do not establish improvement over older runs.

Trace review explains the practical limits:

- Both refactors satisfied independent helper-substitution and preserved-behavior
  checks. The compact attempt recovered from one exact-text mismatch, wrote a
  valid artifact and ran a passing visible check, but never produced a final
  answer within eight iterations.
- Both positive-total patches passed acceptance and preserved protected legacy
  bytes. Each recovered from an escaped-newline edit mismatch. Minimal ran a
  failing focused check before the final write and did not recheck, leaving stale
  evidence. Compact saw the known broad failure before editing, then passed only
  the focused check afterward; it neither reran the broad suite nor finished an
  answer. A passed latest focused check does not erase the known broad failure.
- Both feature patches updated both modules but failed four independent cases:
  very long leading-zero inputs (Python's integer-string conversion limit) and
  non-ASCII decimal digits accepted by `isdigit()`. Compact passed the visible
  check despite these defects; minimal ran no check. File count and visible
  check success do not demonstrate task acceptance.
- Four normalized blank final replies triggered recovery across the six traces.
  This is observed reply/recovery behavior, not evidence about hidden reasoning
  or its backend cause. Five attempts emitted a plan despite the gate being off;
  bookkeeping consumed tools/iterations, without proving a causal remedy.

Post-hoc replay applied each saved diff unchanged to a fresh copy of its original
fixture and reproduced all six acceptance exit codes. Original source, criteria,
references and raw results remained unchanged. The two feature diagnostics only
raised bounded output capture from 12,000 to 200,000 bytes, using the same owned
bootstrap and criteria: each reports 4 failed, 24 passed. This exposes failure
summaries obscured by long parameter IDs in the original bounded output; it is
not new inference, a model retry, a repaired patch or a changed score.

To reproduce inference, extract the frozen recipe to an absolute temporary path,
prepare an isolated **clean** checkout of the pinned harness revision and its
normal dependencies, and install the exact model digest on local Ollama. From
that checkout run (substitute the extracted recipe path):

```sh
uv run python -c 'import runpy; runpy.run_path("/absolute/tmp/reproduce.py", run_name="__main__")' --report /absolute/new-directory/report.json
```

The archived recipe is Python-AST-equivalent to the executed script, with only
formatting/comments changed. It refuses a changed/dirty harness, a different
model digest or an existing report/metadata path. The PR's newer checkout is
intentionally rejected. Backend/hardware nondeterminism can still change results.
For artifact-only replay, copy the fixture, apply its row's unified `diff` with
`git apply`, then use `evals.real_run.acceptance_check` on that disposable copy;
no model is required.

The model was initially unloaded and explicitly unloaded afterward; a subsequent
`ollama ps` was empty. The pre-existing Ollama server remained running. No paid
calls, additional models, Duodoro edits or manual rescues occurred. Larger repeated
and held-out comparisons, repository questions and ten real-project workflows
remain open; this six-attempt development run does not complete Milestone 6.

Validation of the archival PR: 99 focused tests and 1,205 offline tests passed
(five live tests deselected), plus Ruff, formatting and mypy. Both scripted matrices
passed 90/90 escalation decisions; `scripted-contract` labels and metric limits
were inspected. The existing Chroma deprecation warning remains. These offline
checks validate mechanics and packaging, separately from the six live observations.

## 2026-10-18 — Eight versus twelve model iterations (local development)

Six fixed local Gemma attempts compared explicit named profiles with eight or
twelve iterations at clean revision `f25c7e0eb7755effa017f09c15910dd7730cd31c`.
Each cap produced **2/3 accepted artifacts**. Completed-and-accepted counts were
**0/3 at eight, 2/3 at twelve**. This small public development comparison establishes
no repeatable coding-quality gain, and the global default remains eight.

Evidence:

- [Eight-iteration raw report](results/2026-10-18-gemma4-12b-iterations-8.json)
- [Twelve-iteration raw report](results/2026-10-18-gemma4-12b-iterations-12.json)
- [Frozen settings, profiles, order, hashes and cleanup](results/2026-10-18-gemma4-12b-iterations.metadata.json)
- [Exact executed reproduction recipe](results/2026-10-18-gemma4-12b-iterations-reproduce.py)
- [Unchanged saved-diff replay audit](results/2026-10-18-gemma4-12b-iterations.replay-audit.json)

The model was `gemma4:12b`, digest
`4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`,
Ollama 0.33.2 on an Apple M1 Pro with 16 GiB RAM. Requests explicitly used
16,384 context, thinking off and temperature 0; after-attempt active-model
observations reported the same digest/context. This is not per-request backend
or tokenizer attestation. Both variants used the shared application runtime,
full tools with canonical memory names, compact `coding-v3-compact-v1`, disposable
trusted fixtures and `owned-pytest-v1` independent acceptance. Code retains host
permissions; disposable copies are not process isolation.

The selected profile name was `local` in both groups. Only its iteration value
and resulting definition hash differed. All other recorded runtime settings and
tool schemas matched for each task pair. Fixed limits were 24 tools, 180 seconds
for dispatch, 60 seconds per request, 8,192 turn output tokens, 2,048 requested
output tokens per request, 64,000 total tokens, 16,384 effective context, 256 minimum
output tokens and 3 estimated bytes/token. One completion recovery, two identical
calls and four touched files were allowed. Plan/check requirements, root instruction
loading and support tools were off; project check argv was unset. The observed
check status therefore uses the recognized-command heuristic. Explicit defaults
plus frozen overrides disabled dotenv/environment-derived settings; fact stores
were isolated and unseeded. Token gates are estimate-based and dispatch deadlines
are not hard end-to-end return bounds.

All six planned attempts are retained in execution order below. There was one
attempt per task/cap, no attempt reruns, tuning or manual patch repair. Reports stay separate
by cap so their aggregates do not combine unlike budgets.

| Task | Cap | Artifact accepted | Completion | Observed check | Seconds | Tools | Reported prompt / output tokens |
|---|---:|---|---|---|---:|---:|---:|
| `extract_label_normalizer` | 8 | yes | iteration limit | passed | 114.7 | 8 | 21,104 / 924 |
| `extract_label_normalizer` | 12 | yes | completed | passed | 108.6 | 7 | 21,194 / 857 |
| `positive_total_existing_failure` | 12 | yes | completed | failed | 106.6 | 10 | 30,800 / 678 |
| `positive_total_existing_failure` | 8 | yes | iteration limit | stale | 75.1 | 8 | 19,872 / 452 |
| `retry_after_seconds` | 8 | no | iteration limit | passed | 144.6 | 8 | 22,128 / 1,117 |
| `retry_after_seconds` | 12 | no | wall limit | passed | 180.2 | 10 | unknown |

Trace findings limit the interpretation:

- The twelve-cap refactor completed in **eight responses**, with seven tools. Its
  eight-cap counterpart spent a tool on a rejected no-op replacement and ended
  after eight tools/responses without a final answer. Their action sequences differ,
  so this completion difference cannot be attributed to extra allowed iterations.
- Both bugfix runs had the same first eight tool names/arguments/errors. They
  recovered from a quoted argv and an escaped-newline exact-edit mismatch. At eight,
  the correct edit arrived after failed pre-edit checks and no final answer. At
  twelve, eleven responses allowed post-edit focused pass, broad failure and a final
  answer explicitly naming the untouched legacy date-format failure. Acceptance
  protects the legacy bytes. Completed-and-accepted does not require observed
  passing status: the latest broad check is correctly still failed.
- Both feature patches reject non-ASCII digits but fail the two very long
  leading-zero cases: integer conversion rejects the string while its leading zeros
  are still present.
  Each passed the visible check and failed independent acceptance (2 failed,
  26 passed). The twelve-cap attempt also recovered from malformed/unsupported
  command forms, then reached the 180-second dispatch deadline during its eleventh
  request, after ten responses. Extra iterations did not fix its artifact.
- No blank final replies or completion-recovery messages occurred. Three eight-cap
  runs exhausted iterations; the twelve-cap group had two completed runs and the
  wall stop. The aggregate false-completion count is zero in both groups because
  both completed artifacts passed acceptance, without certifying other passed-check
  states or final-answer claims generally.

The cancelled in-flight wall-limited request makes that attempt's full usage
unknown. Its ten recorded responses retain a known prefix of 31,554 prompt and
1,134 output tokens; these are not full totals or zero-cost evidence. The twelve-cap
median reported total (26,764.5) excludes this attempt and covers only two rows;
the eight-cap median (22,028) covers all three. They are not comparable full-cost
measurements. Median latency was 114.7 versus 108.6 seconds and tools 8 versus 10,
but one sample per cell, cold first eight-cap request, alternating fixed order,
backend/cache variation, disposable path differences and shared-host conditions
prevent a reliable latency or causal-quality conclusion. This is a current paired
budget observation; the prior revision's different tool namespace/profile setup
is not an alias-removal experiment here.

Post-hoc replay applied every saved diff unchanged in a fresh fixture copy and
reproduced all six acceptance exit codes. No criteria, reference, source fixture
or raw report changed. All original and replay acceptance output fit the normal
caps and reported no clipping; short explicit failure IDs retain the two feature
failures and totals. This is offline artifact replay, not new inference or rescue.
The recipe SHA-256 remained
`ee884ec25bf64b279dad931648dc543de6069ce2ff3ee7d344ddc66a54ce7cae`
through execution and archival.

For inference reproduction, extract the recipe to an absolute temporary path,
prepare a clean isolated checkout of the pinned revision with normal dependencies,
and install the exact model digest on local Ollama. From that checkout run:

```sh
uv run python -c 'import runpy; runpy.run_path("/absolute/tmp/reproduce.py", run_name="__main__")' --output-dir /absolute/new-experiment-directory
```

The output directory must not already exist. The recipe rejects dirty/changed
harness revisions or model-digest mismatch, validates references before inference,
hashes tasks/criteria/reference files, saves each cap's report after each attempt
and stops on provider/runtime errors without retry/fallback. The newer archival PR
checkout is intentionally rejected; use the pinned revision. Backend results can
still vary. Artifact-only replay uses a fresh fixture, `git apply` on the row's
saved diff and `evals.real_run.acceptance_check`, without a model.

The run lasted 20:22:24–20:34:37 UTC on October 18. The model started unloaded and
was explicitly unloaded afterward; `ollama ps` was empty. The pre-existing Ollama
server remained running. No paid calls, additional models, Duodoro edits, manual
rescues or runtime/default tuning occurred. Repeated and held-out comparisons,
20–30 tasks, repository questions and ten real-project workflows remain open.

Archival validation: 141 focused tests and 1,226 offline tests passed (five live
tests deselected), plus Ruff, all 240 Python files formatted and mypy. Both scripted
matrices passed 90/90 escalation decisions; contract labels and metric limits were
inspected. The existing Chroma deprecation warning remains. These gates validate
mechanics/packaging, separately from the six live observations above.


## 2026-10-19 — frozen ten-task Duodoro client cohort

[Full report](projects/duodoro/TRIAL_RESULTS.md) and
[machine-readable observations](results/2026-10-19-duodoro-client-trial.json).
Real local Gemma `gemma4:12b` / Ollama 0.33.2, runtime
`cd9891718193ae4d9bb56056ae8aea8e57a9ef4c`, frozen Duodoro source/criteria and
five CLI/five panel initial submissions. Ten tasks plus T03's one reserved fork
produced **2/10 accepted artifacts, 1/10 completed-and-accepted**. The 8/10 target
was missed: five final budget stops, four failed runs without normal response
evidence, one completed/passed run. T02 passed external acceptance despite its
budget stop; T10 completed and passed both checks. Failed runs' missing usage is
unknown, not zero. Six known phase reports total 274,169 prompt / 6,206 output
tokens; five phases are unknown, so these are not full-cohort totals.

Cancel/restart/fork controls ran before any partial edit; T05's planned reversal
was rejected and is not a demonstrated recovery success. No patch repair,
replacement tasks, hidden retries or new model hints occurred. Final artifact
rechecks reproduced all ten grades; scope/hash/settings audits passed. Raw local
evidence and cleanup are documented in the report. The temporary API stopped,
Gemma was unloaded, and the existing Ollama service remained running. Duodoro's
working checkout was untouched; no paid calls or Docker actions occurred.

This single public development cohort is not a same-model baseline comparison,
held-out evaluation or evidence of quality improvement. G2 remains open, G3 was
missed, and G4/G5 remain unmet. This docs/data report did not rerun G0: the frozen
runtime's 1,291 offline tests/static checks/two scripted 90/90 matrices remain
historical mechanics evidence, separate from these live results.

## 2026-10-20 UTC — frozen Gemma exact-edit feedback comparison

[Full report](projects/duodoro/EDIT_FEEDBACK_RESULTS.md) and
[all twelve observations](results/2026-10-20-gemma-edit-feedback-comparison.json).
The reviewed PR #152 freeze ran T01/T04/T09 twice per arm through eight actual
CLI and four panel submissions. Treatment is published runtime
`5974d2201cb12829de2b593b602c649515e98b0d`; local control
`1c9249af2b5bcaad2e1628c130a9e879c934196f` restores only the old mismatch helper.
Reporting commits were not inference runtimes. Both use the same frozen Gemma
digest, Ollama 0.33.2, Node 24.22.0, Python 3.14.5, explicit compact/full profile,
300-second dispatch allowance and unchanged token/tool/iteration limits.

Control produced **0/6 accepted artifacts and 0/6 completed-and-accepted**;
treatment produced **6/6 and 4/6**, respectively. Both treatment duration patches
passed external acceptance but stopped before completion/checks. Five control
attempts exhausted budgets; one blocked on repeated grep before any edit. Seven
CLI response timeouts were resolved by saved-run inspection without resubmission.
All twelve have known reported usage, totalling 616,347 prompt / 14,454 output
tokens; this is not independent backend accounting or an efficiency comparison.

Captured base profiles match, but rendered system prompts include differing frozen
absolute workspace paths with arm names. Two of five comparable tool prefixes
diverge before the changed feedback, and the sixth pair lacks a control mismatch.
Selection after known failures, two repeats, fixed order and one model prevent a
general causal/held-out claim. Original ten-task 2/10 and 1/10 results remain
unchanged; G2's reversal and G3–G5 remain unmet.

Final external rechecks reproduced all twelve saved artifact grades; single-run
storage, profile/settings/criterion hashes and protected-file preservation passed.
The report records four operator lookup/cleanup/probe corrections, no coding
rescues, raw local evidence/scripts and audit hashes. The API stopped and Gemma
was unloaded, preserving the existing Ollama service. No Duodoro working-checkout
edits, paid calls, new models or Docker actions occurred. This docs/data report
does not rerun preparation's G0: treatment's 1,330 offline passes and control's
1,325 common passes (five explicit diagnostic exclusions, plus five live exclusions
in each arm), static checks and both scripted 90/90 matrices per arm remain
historical mechanics evidence. The public report is not a portable full-trace kit.

## 2026-10-20 — offline duration budget diagnosis (no inference)

[Diagnosis](projects/duodoro/DURATION_BUDGET_DIAGNOSIS.md) and
[metadata](results/2026-10-20-duodoro-duration-budget-diagnosis.json) replay both
accepted-but-incomplete T01 treatment runs at pinned runtime `5974d22`. Each has
52,943 reported total tokens used, 11,057 remaining, and a 12,762-token final prompt
heuristic plus a 256 minimum output reservation. The original gate rejects with
the exact saved reason even without reconstructing a reported anchor. The root
instruction result is the largest repeated payload (57.09% of final message bytes),
not a measured percentage of backend tokens. Seven request heuristics and null
per-request reported usage remain distinct. No instruction removal, continuation,
patch repair, calibration, budget change or new model result is implied.

Two offline recipe executions produced identical metadata; runtime/run/settings/
schema hashes matched. Existing context/trace regressions passed 27 tests. This
docs/data investigation did not rerun the full gate/matrices. Original ten-task
and paired-comparison counts, unmet release gates and path-confound limits remain
unchanged. The recipe/raw inputs remain local at the diagnosis's recorded path;
no inference, paid calls, Duodoro working-checkout edits or Docker actions ran.

## 2026-10-20 — frozen same-root Gemma read-batching comparison

[Full report](projects/duodoro/BATCH_READ_RESULTS.md) and
[all four observations](results/2026-10-20-duodoro-batch-read-comparison.json).
Actual CLI T01 submissions used pinned control `45f4182` and local treatment
`1f7e0f6`, which adds only one independent-read-batching sentence. Common literal
workspace paths and the frozen first-request assertion matched before forwarding;
all instruction/tool text and approved limits remained intact.

Control produced **0/2 accepted artifacts, 0/2 completed-and-accepted**; treatment
produced **2/2 accepted, 0/2 completed-and-accepted**. All four made seven requests,
stopped at the total-token estimate gate and ran no check. No read batch appeared
in normalized archived history; raw suppressed/unarchived calls remain outside
that coverage. This is not evidence that batching improved coding or efficiency.
All usage is known: 205,942 prompt / 3,266 output tokens across four attempts.
Every CLI response timed out; saved read-only inspection recovered outcomes without
resubmission. No operator coding rescue, artifact repair or replacement attempt.

Final external regrades reproduced every artifact grade. Runtime/environment,
settings/criteria/reference/staging, first identities, captured profiles, single-run
stores and protected-file audits passed. All four full workspaces are preserved;
the owned API stopped and Gemma was unloaded, retaining the existing Ollama service.
Local evidence/recipes/hashes and limitations are documented in the report. No
Duodoro working-checkout edits, paid calls, model switches or Docker actions.

Keep the production prompt and approved gates. This one selected public task and
two repeats per arm do not establish a general/causal or held-out gain. Original
ten-task and feedback comparison results and G2–G5 remain unchanged/unmet. This
docs/data report did not rerun preparation's two 1,342-test offline gates, static
checks or both scripted 90/90 matrices per arm; those remain mechanics evidence.

## 2026-10-20 — existing editing presets on the same duration task/root

[Full report](projects/duodoro/TOOLSET_RESULTS.md) and
[all four observations](results/2026-10-20-duodoro-toolset-comparison.json).
The reviewed PR #162 cohort used one published runtime `ce5d54c`, original T01
criteria and unchanged budgets. Both actual CLI repeats per compact preset used
the same literal workspace root with frozen first-input assertions. Whole-file
removes only `replace_text` and selects its existing matching edit instruction;
retained schemas and other settings match. All instruction/tool text is retained.

Full: **0/2 accepted artifacts, 0/2 completed-and-accepted**. Whole-file:
**2/2 accepted, 2/2 completed-and-accepted**, with passing configured checks.
Full's seven requests leave the second formatter unfixed after mismatch recovery
and stop at the total-token estimate gate; whole-file's eight requests include one
successful full-content write, the configured check and a final answer. Both
whole-file answers also contain an unexecuted inaccurate Python test illustration;
limited source/tool review records that issue separately from artifact acceptance.
The actual unchanged visible test invokes Node and passed. No complete semantic
answer score is claimed.

All four have known reported usage: **220,580 prompt / 3,901 output tokens**.
Whole-file did more completed work with more reported tokens; no equal-work
efficiency ranking follows. All four CLI responses timed out while owned work
continued; original-key read-only inspection retrieved final evidence without
resubmission. Journal intervals range from 161.67 to 190.77 seconds, separately
from pure inference, operator cleanup or hard return deadlines.

Final external regrades reproduced all outcomes. Frozen runtime/environment,
settings/criteria/reference/staging, first identities/profiles, checkpoint pairs,
single-run stores, archives and protected files passed audit. Entire artifacts,
traces, budget journals, client/server logs, recipes and hashes remain local at
the report's path. Owned APIs exited/listeners stopped and Gemma was unloaded,
retaining the existing Ollama service. No coding rescue, artifact repair, execution
setup correction, Duodoro working-checkout edit, paid call or Docker action.

Keep full as the default; stale-hash/large-write risks remain, and this selected
public task/two repeats is not held-out or general causal evidence. Original
ten-task/feedback/batching counts and G2–G5 remain unchanged/unmet. This docs/data
report did not rerun preparation's shared-runtime 71 focused/1,344 offline tests,
static checks or two scripted 90/90 matrices; those remain historical mechanics
evidence, separate from these real-tool coding observations.

## 2026-10-21 UTC — one bounded live CLI recovery workflow

[Report](projects/duodoro/RECOVERY_RESULTS.md) and
[structured observations](results/2026-10-21-duodoro-recovery-workflow.json).
Runtime `dacf072` used the reviewed PR #166 freeze: original Duodoro formatter
behavior/scope/criteria/reference, explicit source/test-focused request variant,
compact/whole-file and two identical reserved 150-second/32K-total-token phases.
This is not an original-T01 repeat, new baseline cohort or quality comparison.

The source made one real formatter write, then cancelled before check/answer
completion. After actual API restart and CLI review, one guarded fork ran the
configured pytest check and completed. External acceptance passed with two checks,
and retained-artifact regrades reproduced it. Source/scored manifests match: the
fork checked/reported the existing patch rather than repairing an incorrect one.
Source: five requests/four responses, unknown total usage with known prefix
9,592 prompt/439 output. Fork: two requests/two responses, reported 6,806/212.
The combined known response prefix is not an aggregate total or token-compliance
claim. No transfer, third phase, coding hint, patch rescue or replacement attempt.

After grading, a declared later sentinel edit survived eligible source-file
reversal; target bytes/mode returned to baseline. Fork target equality made it
ineligible, so no reversal was submitted for it. Another API restart retained
identical saved runs/archives and keyed reversal outcomes without execution.
Audits cover frozen identity/profile/settings/recipes/criteria/reference, scoped
files, root identity, exactly two runs, unchanged evidence and cleanup. All four
owned API processes exited, listener released and Gemma unloaded; existing service
preserved. The actual grader sources/interpreter match the pinned runtime/environment.
The final answer's formatter/check statements have limited source/tool review,
without a complete semantic rubric or model judge.

Raw artifacts, profiles, journals, transcripts, client/server logs, recipes and
regrades remain under the report's local root. No execution setup correction,
manual patch repair, Duodoro working-checkout edit, paid/new model call or Docker.
The retained initial scripted capture preparation error was not a live attempt.
One public variant/sequence does not establish general recovery, browser workflows,
held-out improvement or release readiness. Original ten-task/comparison counts,
no-go and broader G2–G5 remain unchanged. Docs/data only; preparation's 41 focused,
1,363 offline tests and two scripted 90/90 matrices were not rerun as model evidence.


## 2026-10-22 UTC — frozen whole-file ten-task client cohort

[Report](projects/duodoro/WHOLE_FILE_TEN_TASK_RESULTS.md) and
[structured results](results/2026-10-22-duodoro-whole-file-ten-task.json).
After PR #173 merged, runtime `d1e5778` ran all ten original tasks once through
assigned CLI/panel clients with compact/whole-file, original criteria and limits.
Eleven coding turns yielded **2/10 accepted artifacts, 2/10 completed-and-accepted,
2/10 acceptable with known reported usage within nominal budgets**. T01/T04 pass;
T06 completes/passes visible checks but accepts unsafe integers and fails external
acceptance. Five tasks have provider read timeouts; T03's fork reaches its reserved
150-second dispatch limit; T05 stops for missing output usage. Seven task totals
remain unknown. Known response prefixes are retained separately, never as totals.

All ten external grades reproduced. Frozen identities/profiles and protected-file/
mode audits passed. T03's early cancellation/restart/reviews/fork preserved the
source run but produced no patch; T05's unchanged target was ineligible for reversal
and its declared later sentinel survived restart. No patch repair, replacement,
paid/new-model call, dependency install, Docker or Duodoro working-checkout edit.
Fifteen owned APIs exited, port released, Gemma unloaded and owned tab closed.
Read-only DOM and actual controls/SSE worked despite snapshot/native-computer-use
timeouts; screenshot coverage is unavailable. Local reporting-helper corrections,
raw artifacts/journals/grades/recipes and hashes are retained in the report.

G3 missed again: retain no-go for this unattended envelope and freeze feature
batches. Broader G2, held-out/repeated/two-model G4 and Jorge's G5 remain open.
Original 2/10 artifact, 1/10 completed results remain unchanged; differing runtime,
preset, paths/client wait and one public sample/task preclude causal improvement
or general quality claims. Docs/data only; preparation's 82 focused/1,363 offline
and two scripted 90/90 matrices remain historical mechanics evidence. Fresh report
checks validated all rows/counts/evidence hashes and reproduced all artifact grades.


## 2026-10-22 UTC — three offline whole-file failure investigations

[Diagnosis](projects/duodoro/WHOLE_FILE_FAILURE_DIAGNOSIS.md) and structured
[timeouts](results/2026-10-22-duodoro-timeout-diagnosis.json),
[missing usage](results/2026-10-22-duodoro-missing-usage-diagnosis.json), and
[safe-integer counterexamples](results/2026-10-22-duodoro-safe-integer-diagnosis.json).
Retained timeout metadata bounds the failure after plan completion, without raw
failed requests/responses or backend timing/root-cause evidence. Synthetic transport
probes validate classification/no-retry only. T05's canned-tool recorded-response
replay matches five request identities/allowances, eleven budget events and saved
messages, reproducing the same incomplete/unknown-usage stop. T06's unchanged
artifact passes the visible NaN check but fails frozen external acceptance and
three unsafe-integer probes in both waiting/ready states; its safe-integer answer
claim is false. These are diagnosis/mechanics and artifact counterexamples, not
new inference, model-quality improvement or evidence that changing limits fixes it.

All 203 original evidence-file hashes match before/after each recipe. Local recipes
and unchanged T06 copy remain at the diagnosis's path. Focused regressions passed
25 provider/failure-review, two missing-usage and 28 criterion cases (55 total).
No application/evaluator code, prompt/default, frozen criterion, model, retry,
timeout or budget changed. No paid calls, services, Docker or Duodoro working-
checkout edits. Original cohort counts/no-go and G2–G5 remain unchanged; minimal
provider diagnostics or a narrower envelope require their own explicit next slice.


## 2026-10-22 UTC — provider diagnostic launch blocked, no inference

[Setup report](projects/duodoro/PROVIDER_DIAGNOSTIC_RESULTS.md) and
[structured audit](results/2026-10-22-duodoro-provider-diagnostic-blocked.json)
retain the two scheduled outcomes at frozen runtime `594ac44`: T05 API startup
failed before CLI submission; T09 was not launched after the same missing-identity
recipe defect was reproduced offline. Zero coding/model requests; acceptance,
completion and usage are unobserved, not two failed model artifacts. Scripted
preflight used a different factory path and missed the live row-selection error.

The frozen recipes/inputs/criteria/references, both initial workspace manifests and
Duodoro HEAD/status/index/refs match. Owned API exited, port released and loaded
models remain empty. No repair, replacement, rerun, dependency install, paid call
or Docker action. Original cohorts/no-go and G2–G5 are unchanged. Correct source-
factory validation and fresh independently reviewed preparation are required
before any future inference. Preparation's 60 focused/1,397 offline/static and
both 90/90 scripted matrices remain historical mechanics evidence, not model quality.


## 2026-10-23 UTC — two-task provider diagnostic observations

[Full report](projects/duodoro/PROVIDER_DIAGNOSTIC_LIVE_RESULTS.md) and
[structured evidence](results/2026-10-23-duodoro-provider-diagnostic.json) retain
T05 then T09 once each through actual CLI at reviewed runtime `a9d6072`.
Both fail with read_timeout before edits: **0/2 accepted artifacts, 0/2 completed-
and-accepted**. External rejection reproduces on unchanged archived copies.
All ten response/failure events retain bounded metadata across restart without
truncation; failed HTTP durations are 60,015.958 and 60,004.900 ms. Aggregate token
usage is unknown; known prefixes 14,761/115 and 13,560/201 are not totals.
All eight returned responses have present usage fields; historical T05 missing
usage did not reproduce. Backend cause, pure inference and wire state remain unknown.

Frozen first identities/profiles/criteria/recipes/preflight evidence and protected
files/modes match; both roots have exactly one run and no reversal. CLI saved
review remains identical after review-only restart. Four owned APIs exited, model
unloaded and existing service/Duodoro HEAD/status/refs/index preserved. No retry,
replacement, coding rescue, execution setup correction, paid/new-model call,
dependency install or Docker action. Historical failed freeze and original cohorts
remain unchanged. Preparation's 78 focused/1,415 offline/static and both 90/90
scripted matrices remain historical mechanics evidence. These selected attempts
validate observed diagnostic coverage, not a remedy or held-out/causal improvement;
retain no-go and open G2–G5.

## 2026-10-24 — Frozen parser edit-preset comparison

[Results](projects/duodoro/PARSER_EDIT_RESULTS.md) and
[structured evidence](results/2026-10-24-duodoro-parser-edit-comparison.json) retain
all eight actual CLI cells at frozen `9f97655`: full 2/4 artifacts and
completed-and-accepted, whole-file 0/4. Both full T08 refactors used two successful
exact edits, passed independent behavior/source review and preserved unrelated
parser behavior. Both whole-file T08 artifacts completed/passed visible checks
but changed the intention whitespace regex and failed independent acceptance.
T02 failed in both arms: full rejected edits then token prevention; whole-file
missing reported usage. Those two aggregate usage totals remain unknown.

All 54 input pairs match saved identities/allowances and offline production
MockTransport serialization. Post-execution regrades, full archives, SQLite/native
GET/archived responses, protected files/modes, source reviews and 228 outcome hashes
pass audit; all sixteen owned APIs exited and Gemma is unloaded. Duodoro Git
fingerprints and immutable preparation evidence match. Reporting-only archive-
wrapper and parent-PATH corrections are retained; final extra regrades use pinned
Node bytes. Original coding/grades already used the managed environment. No model
input/artifact/coding attempt was repaired or repeated for reporting.

Only existing preset plus its necessary instruction differs; CLI is held constant
in both arms rather than reproducing original panel workflows. Same-task/preset
diffs and known token totals match across repeats at temperature zero; variation
is limited. This selected public-task observation supports the bounded refactor
mechanism, not a general/causal/held-out/two-model gain or a new supported envelope.
Historical formatter observations favoring whole-file remain separate. Original
cohort counts, 8/10 target, no-go/feature freeze, defaults and G2–G5 stay unchanged.
No paid/additional model, Duodoro edit/push, dependency install or Docker action.
The report uses frozen G0 (116 focused/1,453 offline, static and both scripted90/90)
plus new outcome audits; it does not claim those mechanics tests as model successes.


## 2026-10-25 UTC — Invalid-hash feedback comparison

[Results](projects/duodoro/HASH_FEEDBACK_RESULTS.md) and
[structured evidence](results/2026-10-25-duodoro-hash-feedback-comparison.json)
retain four original-T02 CLI cells at published e92f6a1/e416ce5: control 0/2 and
treatment 0/2 accepted artifacts or completed-and-accepted. Each read AGENTS.md,
then hit an approximately 180-second second-call read timeout with no edit/check
or invalid-hash feedback exposure. Absent final completion/check fields stay null;
initial session archives are not invented final turns. The diagnostic did not
activate, so its live recovery effect remains unmeasured and backend cause unknown.

All eight request pairs match saved evidence/offline MockTransport serialization;
paired wire prefixes match through both requests, while exact normalized records
differ after the first. Four aggregate token totals remain unknown; each known
1,962-prompt/22-output prefix is not a total. Further regrades reject unchanged
artifacts; SQLite/native GET, full trees/modes, source/recipe/preparation and 104
outcome hashes pass audit. Eight owned APIs exited, model unloaded and Duodoro Git
fingerprints preserved. No replacement, repair, extra turn or execution correction.

Selected public-task low-variation observations establish no feedback gain,
regression, broader envelope or release closure. Existing basic diagnostic stays
usability, not measured quality. Original cohorts/8of10 target/no-go/defaults/G2–G5
remain unchanged. No paid/additional model, dependency install, Duodoro edit/push
or Docker action. Report uses frozen G0 (control 1,453/treatment 1,465 offline,
static and both 90/90 matrices each) plus the settled-outcome audit; no new local
full suite was run for docs/data.
