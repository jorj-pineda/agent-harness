# Preserve mock captures before fresh exact-path staging

This operator step applies only to a complete prepared-workspace capture bundle.
It does not archive or reset model attempts, user repositories or arbitrary work.
Supply the exact index SHA-256 selected for review:

```sh
uv run python -m evals.heldout_staging \
  --directory /tmp/unique-prepared-input-capture \
  --archive /tmp/unique-capture-archive \
  --index-sha256 REVIEWED_INDEX_SHA256
```

Before mutation, verification binds the caller-supplied index digest, original
plan/seal, clean recorded runtime identity, all sixteen schedule rows, exact
normalized/wire file maps, hashes and private file modes. Every captured named
workspace must remain the unchanged original fixture, with matching prepared and
admission receipts and no extra inventory. Recorded roots and both request bodies
must name the original canonical workspace path. Unknown files are inventoried
without reading their contents. Source and archive cannot overlap and must remain
outside Git workspaces; archive creation is exclusive.

The complete source directory moves to `archive/captured`, preserving request
bytes, fixtures and consumed admission markers. The archived bundle is verified
again against its original named roots. Only then is a fresh source directory
created and all sixteen original fixtures prepared and verified at their previous
canonical paths. New copies have no admission marker. `staged.json` is written
last and binds the preserved index hash/archive location; its completion grants
no inference permission. No old admission marker is deleted or reused.

A scripted real-tool regression captures the first staged request through the
production adapter and confirms normalized and wire bytes exactly match their
archived originals. This proves the exercised path reconstruction, not equality
for future changed configurations or launch readiness. Each eventual request still
needs its independently reviewed first-input guard and current preservation gates.

The supplied hash binds bytes, not the identity or authority of an approver. A
separate execution-freeze approval receipt is still required. Checks/rename/staging
are bounded but non-atomic and do not lock external writers. Source and archive
must permit same-filesystem rename. Failures retain both complete or partial owned
directories; no rollback, reset, deletion or automatic retry occurs. Before trying
anything else, inspect the retained state. A missing staged receipt is incomplete,
and a consumed/live result must never be passed off as a no-forward capture.

This step starts no provider and submits no request. Current resource/model pins,
settlement/refusal/launch recipe, independently reviewed exact execution freeze,
parent approval and fresh launch gates remain incomplete.
