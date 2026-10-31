# Explicit evaluation command environments

Python callers of real_run.run_task, validate_task and acceptance_check may supply
command_env. None preserves the historical allowlisted ambient merge. A supplied
mapping selects replacement mode for task tools, initial/reference visible and
acceptance controls, and the final independent grader. Repository-question grading
remains in process. There is no CLI environment selector or automatic live launch.

The mapping is copied and validated before task workspace admission. It requires a
nonempty PATH of absolute entries. Supported keys are PATH, VIRTUAL_ENV, LANG,
LC_ALL, SYSTEMROOT, TMPDIR, HOME and the two pytest startup flags. Values are strings
without NULs, at most 4096 characters each. Unknown keys refuse. The evaluator
forces PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 and PYTHONDONTWRITEBYTECODE=1; callers cannot
weaken those startup protections. Explicit mappings do not import ambient values.

New result runtime_config records command_environment as replacement-v1 plus a
SHA-256 of the effective canonical mapping, or null for default merge mode. It does
not retain raw values. Legacy rows remain untouched/unknown. The hash identifies
configured values, not approval authority, executable contents or actual environment
changes made by child programs. Tests exercise real tools and external grading on
a disposable scripted task; they establish mechanics, not model quality.

This does not change authoritative acceptance inputs or create an environment
manifest, freeze PATH target bytes, pin packages/libraries, isolate host code or
exclude external writers. The named held-out capture/operator still use their
existing defaults until a separately reviewed environment pin is wired through
that whole recipe. Old captures must not be retrofitted or consumed workspaces
reused. Full live operator/refusal/settlement, exact execution freeze, parent
approval and fresh launch gates remain required before inference.
