# Opt-in fixed project check

Set `CODING_TOOLSET=no_plan_check` with a valid `PROJECT_CHECK_ARGV` JSON array.
This experimental mode starts from no_plan and replaces run_command with
run_project_check, whose input is an empty object. It runs the copied configured
command through the same command runner, allowlist, environment, timeout,
cancellation and output bounds. Model-supplied arguments are rejected before
execution. Other code and memory tools remain available; arbitrary command
selection and formal planning are unavailable in this mode.

Full remains the default; full, whole_file and no_plan behavior is unchanged.
Missing configured checks and required planning refuse during Settings validation.
This changes the interface, not the verification requirement: only an actual
successful check after the latest edit supports verified completion. The tool
returns the actual argv and exit/output evidence; Python3 uses the existing Python
normalization. A later edit or interrupted command invalidates earlier proof.

The shared API, real-task and simulated-tool registries use the same selection.
Prompts identify the available tool, configuration/recovery review uses the same
framing, and a separate prompt version plus captured settings/schema/check hashes
identify the choice. Existing histories remain unchanged and configuration drift
requires review. This is not process isolation or permission to execute source.

This seam targets observed quoted-argv refusals in bounded development traces.
Scripted and real-file tests establish mechanics, not improved model completion.
The retired experiment protocol still selects full/no_plan; a new bounded capture,
reviewed execution freeze, actual parent receipt and fresh gates must precede any
new model attempt. No retry of a closed attempt or automatic default adoption.
