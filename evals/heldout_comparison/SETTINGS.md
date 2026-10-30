# Validated Settings pins

The protocol validator, no-forward capture and both scripted rehearsals now use
one Settings factory. Only explicit initialization values and defaults populate
its fields; environment, .env and file-secret sources cannot override them. It
selects Ollama and the fixed owned loopback origin, model and reviewed budgets.
Ordinary application Settings/environment behavior is unchanged.

New captures retain settings.json (mode 0600, at most 64 KB) and bind its hash in
the completion index. Canonical JSON contains every explicitly classified public
Settings field, including fields that do not appear in the serialized first
request. All credential fields must be unset and are excluded. Model profiles and
hosted endpoint configuration refuse. A schema change, including a new potential
secret field, computed field or populated extra field, refuses before serialization until the classification is reviewed.
Errors do not echo credential values. Pin loading checks regular-file/type, size,
mode, SHA-256 and the typed wrapper without reconfiguring Settings from its data.

Capture checks the pinned Settings before each cell and provider request, after
each result and before its completion index. Drift preserves partial artifacts
without a completion index.

The named operator builds its actual Settings independently and compares the
canonical snapshot to the selected pin before creating evidence/provider resources
or admitting a workspace. It rechecks before each cell and wraps the provider to
check that same Settings object before every delegate request. A request-boundary
failure latches closed even if values are restored. Later cells stop under the
existing cohort policy, retaining their unrun/no-result status. Provider closure
still runs; this adds no resubmission or recovery path. The existing first-input
and result/configuration/preservation checks remain additional evidence.

The selected capture hash binds bytes, not approver authority. Staging preserves
and verifies the settings file with the rest of the capture. Capture indexes
without the optional settings hash remain readable for historical preservation;
the named operator rejects them before allocation/admission. Never retrofit a
new pin into old retained captures or reuse their consumed workspaces. Generate a
fresh capture, stage and rehearsal at new locations instead. Historical outcomes
and evidence bytes are not rewritten.

“Validated Settings” means resolved Pydantic field values, not a complete process
or environment freeze. Relative paths remain relative values, and unused default
provider/data fields are still included. Session workspace/inspection policy,
registries, temporary fact stores, command PATH/environment, installed packages,
executable bytes, endpoint/model/resource ownership and backend compliance need
their own pins and gates. The immutable expected bytes do not lock a mutable
Settings object or exclude external writers between checks. A missing/failed
final report is not preserved completion evidence. No process isolation, full
execution freeze, launch approval, inference or model-quality claim follows.
