# One-shot fresh candidate preparation

`prepare_fresh_candidate` creates a new twelve-cell candidate using an explicitly
selected existing interpreter, clean retained source, runtime/dependency files,
model store and command environment. It never installs dependencies, resolves a
new environment, starts a backend or issues a parent receipt.

The synchronous API requires the exact running Python entrypoint outside the
retained source tree, with no local environment in that tree. Selected inputs must
not overlap the new evaluation-owned output. Existing output is refused. Runtime
and dependency observations bracket their preservation; a separately verified image
is copied using the existing image helper. Generation stays in the existing internal
mock capture driver. All twelve consumed captures are archived before fresh weak/
stronger fixtures are staged at their exact captured paths.

The resulting distinct manifest binds those artifacts, environment, executable,
image and future output path. Read-only fresh preflight must pass before manifest
publication; source/interpreter checks precede the final private preparation
receipt. Failures retain partial state, write withdrawal/failure markers and never
reset or retry. Candidate and rehearsal bundles must be separate: later consuming
a rehearsal does not admit or change a candidate.

This is preparation, not authority or an executable launch recipe. Twelve staged
cells remain unadmitted and the future execution output does not exist. Scripted
reference successes establish mechanics only; runtime/dependency selection is not
complete import closure or host isolation. A concrete retained freeze, candidate
checks, actual approval and fresh launch checks remain before real inference.
