# Read-only runtime inventory

Run with the managed interpreter and explicit installed executable paths:

```sh
uv run python -m evals.heldout_runtime_inventory \
  --executable python=/absolute/path/to/managed/python \
  --executable git=/absolute/path/to/git \
  --report /tmp/unique-runtime-inventory.json
```

The selected Python must resolve to the running interpreter. Up to sixteen
explicit absolute paths are resolved and regular executable files hashed in
bounded streamed reads with the existing before/after identity checks. Labels,
file sizes and package metadata counts/lengths are bounded. Installed distribution
names are normalized; duplicates refuse. No selected executable is run, package
imported, service started or environment/credential file inspected. The private
report is exclusive and outside Git workspaces; occupied paths are preserved.

This inventories interpreter bytes, selected executable bytes, Python version and
installed distribution names/versions. It does NOT hash package source, shared
libraries or interpreter standard-library files, and does not freeze PATH, command
environment, symlink targets after observation or external writers. Metadata
versions are not proof of dependency bytes. The flags for dependency-byte and
command-environment verification remain false. Reports are not signed approvals,
atomic snapshots or launch gates. A retained runtime, dependency/environment pins,
full live operator/refusal/settlement recipe, exact reviewed execution freeze,
parent approval and fresh launch gates remain required before inference. No model
quality or release claim follows.

## Read-only recheck

```sh
uv run python -m evals.heldout_runtime_guard \
  --inventory /tmp/selected-runtime-inventory.json \
  --sha256 SELECTED_REPORT_SHA256
```

The recheck binds the supplied report hash and private mode before parsing a
strict bounded schema or observing runtime files. It compares a fresh observation
of the actual running interpreter, selected executable paths/targets/bytes/modes,
Python identity and installed distribution metadata with the selected inventory.
Report bytes/mode are checked again afterward. Mismatch refuses without rewriting
or rebinding the inventory. This utility constructs no provider and runs no selected
executable. The hash selects bytes, not approval authority; an arbitrary report
must not be treated as a trusted freeze. Host code remains trusted.

These observations are non-atomic and do not exclude external writers or establish
later freshness. Package versions still do not verify dependency/library bytes.
The recheck is not yet wired into capture/operator dispatch and does not authorize
inference. Retained runtime/dependency/resource pins, the full live operator and
exact freeze/approval/fresh-launch gates remain incomplete.
