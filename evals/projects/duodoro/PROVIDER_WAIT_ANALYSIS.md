# Offline provider-timeout analysis

No inference, service/debug change or task replay occurred. This reads the retained
[one-attempt input diagnostic](REQUEST_INPUT_RESULTS.md), constructs configured
providers without calling chat, and extracts fixed-format metadata from existing
shared Ollama logs. Original outcomes, private inputs and freeze hashes match.
[Structured observations](../../results/2026-10-23-duodoro-provider-wait-offline.json)
retain the bounded recipe hash, configuration and source-prefix identities.

## Why test the existing timeout setting

The retained run's plan ended about **70.275 seconds** after its UTC run creation;
run failure was recorded at about **130.361 seconds**. Its fifth adapter call
reported `read_timeout` after **60,046.100 ms**. The configured turn dispatch
allowance was 300 seconds, and no wall-budget stop was reported.

Subtracting that run age from 300 gives **229.725 seconds nominally** at plan end.
This is not a measurement of the actual monotonic deadline remaining: run timestamps,
setup and finalization have different scopes. It only motivates testing whether
the separate provider timeout cuts off a potentially useful response while the
unchanged five-minute task allowance remains available. A longer wait could still
stall, fail or consume that allowance without producing an acceptable patch.

The inspected production factory passes `request_timeout_seconds` to the Ollama
HTTP client; it is not a prompt/tool or generation-token setting. Construction
without chat confirms that setting it to 180 changes **connect, read, write and
pool** from 60 to 180 seconds. All other validated settings, including 300-second
turn dispatch and token/tool limits, remain equal. This is a provider-timeout scalar
experiment, not a read-only timeout override. HTTPX documents those four separate
phases and read inactivity per received chunk. [HTTPX timeout documentation](https://www.python-httpx.org/advanced/timeouts/).

## Existing shared-log candidates

Ollama documents the macOS server log at `~/.ollama/logs/server.log`.
[Official troubleshooting](https://docs.ollama.com/troubleshooting),
[macOS log locations](https://docs.ollama.com/macos).
The one-off reader scans only regular `server*.log` files, with caps of 16 MiB
combined, 8 MiB per file and 4 KiB per processed line. Only fixed-format POST
`/api/chat` access metadata is emitted; other payloads, IPs, URLs, headers,
configuration, prompts and error text are not copied or published. Eight duration
parser fixture checks passed. The actual scan covered 3,897,549 bytes without a
file-prefix cutoff or concurrent file change.

Within the retained run window plus two seconds, assuming **America/Chicago** for
the access log's timezone-free timestamps, five candidate records appear:

| Candidate end, UTC under that assumption | Logged status | Reported duration literal |
|---|---|---|
| 21:49:02 | 200 | 21.724184708s |
| 21:49:04 | 200 | 1.424909583s |
| 21:49:29 | 200 | 25.607255792s |
| 21:49:50 | 200 | 21.139355416s |
| 21:50:50 | 500 | 1m0s |

All dates are 2026-10-23. These are **time-window candidates**, not request-ID
correlation. The log is shared; no independent per-request linkage, generation
progress or failed output was retained. The final duration literal may be rounded/
truncated; it is not an exact independent 60,000 ms measurement. Neither status500
nor its timing proves whether the client cancellation, backend generation, queue,
serialization or another cause produced the failure. No raw shared-log snapshot
was copied. Prefix hashes/lengths identify what was scanned; rotation/overwrite
can prevent later reinspection, so these logs are supporting observations only.

Local recipe and sanitized metadata remain at:

`/Users/jorge/.local/share/agent-harness/trials/duodoro-provider-wait-offline-vw4m3xgy`

## Disposition

The new [comparison protocol](PROVIDER_WAIT_PROTOCOL.md) explicitly varies the
existing scalar between 60 and 180 while preserving the user-selected five-minute
per-task dispatch, token/tool/iteration ceilings, original acceptance and model.
It does not infer a fix from the logs, raise the global task budget or change a
production default. Actual same-root inputs/client preflight, exact G0 and an
independently reviewed execution freeze remain required before inference.

This offline docs/data slice did not rerun the application test suite or scripted
matrices: no runtime/evaluator/scorer code changed. Source/recipe/prior-evidence
hashes, parser fixtures and no-chat production construction were checked. Original
cohorts/no-go and G2–G5 remain unchanged; the comparison may return another failure.
