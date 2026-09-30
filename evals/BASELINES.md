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

