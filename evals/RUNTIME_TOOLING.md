# Development tooling beside a retained runtime

Use the development checkout for development commands. Retained runtime trees are
inputs to an exact execution freeze, including paths that ordinary Git/Ruff tools
might ignore. A new cache is drift even when no original source file changes.
The closed token-recovery cohort remains closed; do not clean its cache or resume it.

For formatting an evaluation audit script, run from the development checkout:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m evals.runtime_tooling \
  --runtime /absolute/retained/runtime \
  --working-directory /absolute/external/tooling-directory \
  /absolute/external/audit.py
```

The installed interpreter runs Ruff with `--no-cache`, `RUFF_NO_CACHE=1` and disabled
bytecode writes. Both the explicit working directory and resolved script target
must be outside the supplied runtime, including symlink targets. The runtime and
working directory must already exist. Unsafe paths refuse before Ruff is spawned;
Ruff's output and exit status are returned without retries or installations.
Existing cache/evidence files are never removed. Exact runtime fingerprint gates
remain mandatory and unchanged.

This is an opt-in development helper for the known formatting mistake, not a
sandbox, filesystem write lock or universal tool runner. It cannot prevent other
commands/external writers, control every user Ruff configuration, or make a Python
interpreter launched inside a frozen tree safe before this helper is imported.
Keep that launch outside the runtime as well. It changes no coding-model prompt,
budget, application tool or acceptance criterion, and supplies no quality evidence.
