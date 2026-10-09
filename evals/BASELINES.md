# Real-task baselines

Real coding evaluations: a live model uses the application runtime and real
tools on disposable copies of the five trusted fixtures in `evals/real_tasks/`.
Acceptance tests that the model cannot edit grade the final files. These results
say nothing about other models, tasks, or languages. Scripted-contract and
simulated-tool results live elsewhere and are not comparable.

Raw reports (traces, diffs, acceptance output, usage, runtime config) are in
`evals/results/`.

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
