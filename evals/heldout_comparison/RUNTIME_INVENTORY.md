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
