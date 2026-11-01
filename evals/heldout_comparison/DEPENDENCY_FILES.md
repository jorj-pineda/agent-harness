# Explicit dependency code/library byte pins

```sh
uv run python -m evals.heldout_dependency_files \
  --file /absolute/path/to/reviewed/module.py \
  --file /absolute/path/to/reviewed/extension.so \
  --report /tmp/new-selected-dependency-files.json
uv run python -m evals.heldout_dependency_files \
  --report /tmp/new-selected-dependency-files.json --verify-sha256 SELECTED_SHA256
```

Selection is explicit, not inferred from installed distribution metadata. This
reads only trusted selected Python/code/library files (.py, .pyc, .so, .dylib,
.dll, .pyd, .pth). Final links, nonregular files, .env/.env.* or Git metadata
paths, duplicate canonical paths and count/size/aggregate overflow refuse. Parent
aliases are resolved and recorded as canonical paths. Each file has streamed
exact-length hash and observed before/after identity checks. Up to 4096 selected
files, 256 MB each and 2 GB total are permitted. Reports are private/exclusive,
outside Git workspaces, bounded at 2 MB and retain only paths/hashes/sizes/modes.
No file contents are printed, imported, executed or copied. No downloads occur.

Rechecks bind the selected report hash/private mode and typed schema, refuse
changed path targets or selected bytes/modes, and recheck report identity afterward.
They never rewrite or rebind the manifest. Failures preserve the report. Inputs
are trusted operator selections; path checks and hashing are not process isolation,
an atomic snapshot, external-writer exclusion or later freshness guarantees.

Import closure remains explicitly unverified. These pins cannot establish that
all imported source, bytecode, standard-library, transitive shared-library or data
files were selected. A .pth hash does not execute or resolve its import statements.
No package or library discovery occurs. Selected-file equality must never promote
the runtime inventory's dependency_bytes_verified flag to true or imply launch
readiness. The manifest is not yet wired into capture/dispatch. Complete retained
runtime/dependency/resource pins, the full live operator/refusal/settlement recipe,
exact execution freeze, parent approval and fresh launch gates remain required.
No inference, model-quality or release claim follows.
