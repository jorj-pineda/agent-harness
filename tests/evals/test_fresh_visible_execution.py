import hashlib
import json
from pathlib import Path

import pytest

import evals.fresh_visible_execution_manifest as binding
import evals.fresh_visible_execution_preflight as gate
from evals.fresh_visible_staging import stage_fresh_capture
from evals.fresh_visible_workspaces import SELECTION_SHA256
from evals.heldout_inputs import _save
from evals.real_run import _task_command_environment
from tests.evals import test_fresh_visible_staging as staging_tests

captured = staging_tests.captured
source = staging_tests.source


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def values(root):
    return {
        "format": "fresh-visible-execution-v1",
        "runtime_commit": "a" * 40,
        "selection_sha256": SELECTION_SHA256,
        "inference_authorized": False,
        **dict.fromkeys(
            (
                "capture_index_sha256",
                "settings_sha256",
                "command_environment_sha256",
                "runtime_inventory_sha256",
                "dependency_files_sha256",
                "image_receipt_sha256",
                "binary_sha256",
            ),
            "0" * 64,
        ),
        **{
            key: str(root / key)
            for key in (
                "staged_directory",
                "archive_directory",
                "output_directory",
                "image_directory",
                "ollama_binary",
            )
        },
        "command_environment": _task_command_environment({"PATH": "/usr/bin:/bin", "LC_ALL": "C"}),
    }


def selection(tmp_path):
    manifest = values(tmp_path)
    mp = tmp_path / "manifest.json"
    mh = _save(mp, json.dumps(manifest).encode())
    approval = {
        "format": "fresh-visible-parent-approval-v1",
        "approval_id": "scripted-not-authority",
        "manifest_sha256": mh,
        "runtime_commit": "a" * 40,
        "freeze_reviewed_head": "b" * 40,
        "freeze_merge_commit": "c" * 40,
        "exact_head_review_passed": True,
        "ci_passed": True,
        "merged": True,
        "inference_authorized": True,
    }
    ap = tmp_path / "approval.json"
    ah = _save(ap, json.dumps(approval).encode())
    return mp, mh, ap, ah


def load(selection):
    mp, mh, ap, ah = selection
    return binding.load_fresh_execution(
        mp, manifest_sha256=mh, approval_path=ap, approval_sha256=ah
    )


def test_selected_manifest_and_structural_receipt_bind_read_only(tmp_path):
    selected = selection(tmp_path)
    manifest, approval = load(selected)
    assert approval.manifest_sha256 == selected[1]
    assert manifest.keep_alive_seconds == 300
    assert (
        manifest.coding_prompt_profile == "direct" and manifest.coding_toolset == "whole_file_check"
    )
    assert not Path(manifest.output_directory).exists()
    with pytest.raises(ValueError):
        from evals.heldout_execution_manifest import ExecutionManifest

        ExecutionManifest.model_validate(manifest.model_dump())


@pytest.mark.parametrize(
    "key,value",
    [
        ("selection_sha256", "0" * 64),
        ("format", "heldout-execution-manifest-v1"),
        ("coding_toolset", "full"),
        ("coding_prompt_profile", "compact"),
        ("keep_alive_seconds", 0),
        ("keep_alive_seconds", True),
        ("keep_alive_seconds", "300"),
        ("inference_authorized", True),
        ("output_directory", "relative"),
        ("residency_feasibility_ordinal", 2),
    ],
)
def test_schema_refuses_other_scope_or_implicit_authority(tmp_path, key, value):
    raw = values(tmp_path)
    raw[key] = value
    with pytest.raises(ValueError):
        binding.FreshExecutionManifest.model_validate(raw)
    assert not (tmp_path / "output_directory").exists()


@pytest.mark.parametrize(
    "defect", ["hash", "mode", "runtime", "manifest", "false", "legacy", "source-drift"]
)
def test_receipt_selection_refusals(tmp_path, monkeypatch, defect):
    mp, mh, ap, ah = selection(tmp_path)
    body = json.loads(ap.read_bytes())
    if defect == "hash":
        ah = "0" * 64
    elif defect == "mode":
        ap.chmod(0o644)
    elif defect == "source-drift":
        original = binding._selected_bytes
        calls = []

        def read(*args, **kwargs):
            raw = original(*args, **kwargs)
            calls.append(True)
            if len(calls) == 2:
                mp.write_bytes(mp.read_bytes() + b" ")
            return raw

        monkeypatch.setattr(binding, "_selected_bytes", read)
    else:
        key, value = {
            "runtime": ("runtime_commit", "d" * 40),
            "manifest": ("manifest_sha256", "0" * 64),
            "false": ("ci_passed", False),
            "legacy": ("format", "heldout-parent-approval-v1"),
        }[defect]
        body[key] = value
        ap.write_text(json.dumps(body))
        ah = sha(ap)
    with pytest.raises(ValueError):
        load((mp, mh, ap, ah))
    assert not (tmp_path / "output_directory").exists()


@pytest.fixture
def prepared(source, tmp_path, monkeypatch):
    # Restoring only the synthetic dependency file keeps cases independent.
    (source.parent / "dependency.py").write_text("value = 1\n")
    digest = sha(source / "index.json")
    archive = tmp_path / "archive"
    stage_fresh_capture(source, archive, expected_index_sha256=digest)
    from evals.fresh_visible_staging import verify_staged_fresh

    index = verify_staged_fresh(source, archive, expected_index_sha256=digest)
    image = tmp_path / "image"
    image.mkdir()
    _save(image / "prepared-image.json", b"scripted-image-receipt")
    binary = tmp_path / "ollama"
    binary.write_bytes(b"scripted-executable-not-run")
    binary.chmod(0o700)
    monkeypatch.setattr(gate, "_prepared_models", lambda path: (path / "models", {}))

    def runtime(expected):
        if expected != {"commit": "a" * 40, "dirty": False}:
            raise ValueError("scripted runtime mismatch")

    monkeypatch.setattr(gate, "_runtime_matches", runtime)
    environment = json.loads((archive / "captured/command-environment.json").read_bytes())["values"]
    data = values(tmp_path)
    data.update(
        staged_directory=str(source),
        archive_directory=str(archive),
        image_directory=str(image),
        ollama_binary=str(binary),
        capture_index_sha256=digest,
        image_receipt_sha256=sha(image / "prepared-image.json"),
        binary_sha256=sha(binary),
        command_environment=environment,
        **{
            key: getattr(index, key)
            for key in (
                "settings_sha256",
                "command_environment_sha256",
                "runtime_inventory_sha256",
                "dependency_files_sha256",
            )
        },
    )
    return binding.FreshExecutionManifest.model_validate(data)


def test_fresh_public_pins_pass_without_output_or_cell_admission(prepared):
    index = gate.preflight_fresh(prepared)
    assert len(index.cells) == 12
    assert not Path(prepared.output_directory).exists()
    assert not list(Path(prepared.staged_directory).glob("cell-*/admitted.json"))


@pytest.mark.parametrize(
    "defect",
    [
        "settings",
        "runtime",
        "dependency",
        "environment",
        "receipt",
        "binary",
        "admitted",
        "selected-file",
        "runtime-commit",
    ],
)
def test_semantic_drift_refuses_without_resources(prepared, defect):
    updates = {}
    if defect in {"settings", "runtime", "dependency"}:
        field = {
            "settings": "settings_sha256",
            "runtime": "runtime_inventory_sha256",
            "dependency": "dependency_files_sha256",
        }[defect]
        updates[field] = "0" * 64
    elif defect == "environment":
        updates["command_environment"] = {**prepared.command_environment, "LC_ALL": "POSIX"}
    elif defect == "receipt":
        Path(prepared.image_directory, "prepared-image.json").write_bytes(b"changed")
    elif defect == "binary":
        Path(prepared.ollama_binary).write_bytes(b"changed")
    elif defect == "admitted":
        Path(prepared.staged_directory, "cell-12/admitted.json").write_text("occupied")
    elif defect == "selected-file":
        Path(prepared.staged_directory).parent.joinpath("dependency.py").write_text("changed")
    else:
        updates["runtime_commit"] = "d" * 40
    with pytest.raises(ValueError):
        gate.preflight_fresh(prepared.model_copy(update=updates))
    assert not Path(prepared.output_directory).exists()
    assert not Path(prepared.staged_directory, "cell-01/admitted.json").exists()


@pytest.mark.parametrize("defect", ["occupied", "overlap", "alias", "binary-link"])
def test_path_refusal_precedes_artifact_inspection(prepared, monkeypatch, defect):
    updates = {}
    root = Path(prepared.output_directory).parent
    if defect == "occupied":
        Path(prepared.output_directory).mkdir()
    elif defect == "overlap":
        updates["output_directory"] = prepared.archive_directory
    elif defect == "alias":
        (root / "alias").symlink_to(prepared.staged_directory, target_is_directory=True)
        updates["output_directory"] = str(root / "alias/output")
    else:
        (root / "binary-link").symlink_to(prepared.ollama_binary)
        updates["ollama_binary"] = str(root / "binary-link")
    monkeypatch.setattr(gate, "verify_staged_fresh", lambda *a, **k: pytest.fail("read artifacts"))
    with pytest.raises(ValueError):
        gate.preflight_fresh(prepared.model_copy(update=updates))


def test_late_runtime_or_staged_drift_refuses(prepared, monkeypatch):
    checks = []

    def runtime(expected):
        checks.append(expected)
        if len(checks) == 2:
            Path(prepared.staged_directory, "cell-01/workspace/extra").write_text("drift")

    monkeypatch.setattr(gate, "_runtime_matches", runtime)
    with pytest.raises(ValueError):
        gate.preflight_fresh(prepared)
    assert not Path(prepared.output_directory).exists()


@pytest.mark.parametrize("defect", ["credentials", "flags", "constructed-policy"])
def test_environment_and_constructed_scope_refuse_before_artifacts(tmp_path, monkeypatch, defect):
    raw = values(tmp_path)
    if defect == "credentials":
        raw["command_environment"]["API_KEY"] = "private-sentinel"
    elif defect == "flags":
        raw["command_environment"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "0"
    else:
        raw["keep_alive_seconds"] = 0
    monkeypatch.setattr(gate, "fresh_execution_paths", lambda *a: pytest.fail("inspected paths"))
    with pytest.raises(ValueError) as caught:
        gate.preflight_fresh(binding.FreshExecutionManifest.model_construct(**raw))
    assert "private-sentinel" not in str(caught.value)


def test_different_python_entrypoint_refuses_even_if_resolved_binary_matches(
    prepared, tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    alias = tmp_path / "other-python"
    alias.symlink_to(sys.executable)
    assert alias.resolve() == Path(sys.executable).resolve()
    monkeypatch.setattr(gate, "sys", SimpleNamespace(executable=str(alias)))
    with pytest.raises(ValueError, match="exact selected Python"):
        gate.preflight_fresh(prepared)
    assert not Path(prepared.output_directory).exists()


def select(prepared, **updates):
    kwargs = {
        "expected_index_sha256": prepared.capture_index_sha256,
        "expected_runtime_commit": prepared.runtime_commit,
        "image_directory": Path(prepared.image_directory),
        "image_receipt_sha256": prepared.image_receipt_sha256,
        "ollama_binary": Path(prepared.ollama_binary),
        "binary_sha256": prepared.binary_sha256,
        "output_directory": Path(prepared.output_directory),
    }
    return gate.select_fresh_execution(
        Path(prepared.staged_directory), Path(prepared.archive_directory), **(kwargs | updates)
    )


def test_selection_is_read_only_and_preserves_every_explicit_pin(prepared):
    selected = select(prepared)
    assert selected == prepared
    assert not selected.inference_authorized and not Path(selected.output_directory).exists()
    assert not list(Path(selected.staged_directory).glob("cell-*/admitted.json"))


def test_selection_requires_independently_expected_image_pin(prepared):
    with pytest.raises(ValueError):
        select(prepared, image_receipt_sha256="0" * 64)
    assert not Path(prepared.output_directory).exists()
