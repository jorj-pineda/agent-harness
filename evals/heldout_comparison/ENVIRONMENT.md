# Local model metadata observation

```sh
uv run python -m evals.heldout_environment \
  --base-url http://127.0.0.1:11434 \
  --report /tmp/unique-model-metadata.json
```

The utility sends only bounded GET requests to `/api/version` and `/api/tags`.
Origins must use HTTP and a literal loopback IP; credentials, paths, queries,
fragments, public hosts and hostname resolution are rejected. Redirects and
ambient HTTP proxy configuration are disabled. A private exclusive report retains
only the selected model's observed version/digest and comparison with the fixed
protocol. Other installed model rows are not copied into the artifact.

It starts/stops no service, loads/unloads no model and sends no generation,
embedding, warmup or download requests. An unavailable service or invalid metadata
produces an explicit unavailable record containing only the exception class, then
a nonzero CLI exit. A metadata mismatch also exits nonzero; absent values remain
unknown. Occupied report paths refuse before requests, and task-input overlap
checks are reused. Reports are observations, not mutable pass receipts.

A matching observation does not verify process ownership, model-file hashes,
backend compliance, absence of external writers or remote generation settlement.
Metadata GETs are not an atomic server snapshot. The captured timestamp does not
provide a freshness guarantee for any later launch. Current runtime/owned model
image, explicit resource ownership and fresh launch/settlement gates must still be
verified under the independently reviewed execution freeze and approval receipt.

The first real observation for this slice found no listener on the expected local
port and recorded ConnectError. No service was started to bypass that failure.
That is an environment/setup limitation, not a product regression or a model
attempt. The offline mock tests separately exercise matching/mismatched identities,
missing models, invalid responses and redirect/oversize refusals.
