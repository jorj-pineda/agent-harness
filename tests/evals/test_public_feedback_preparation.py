import json
import sys
from pathlib import Path

import pytest

from evals import heldout_operator_rehearsal as operator
from evals import heldout_preparation as interpreter
from evals import public_feedback_execution_preflight as gate
from evals import public_feedback_preparation as prep
from evals import public_feedback_rehearsal as inputs
from evals.heldout_inputs import _save


@pytest.fixture
def configured(tmp_path, monkeypatch):
    import httpx

    def forbidden_http(*args, **kwargs):
        pytest.fail("Real HTTP transport constructed during offline preparation")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "__init__", forbidden_http)
    root = tmp_path.resolve()
    runtime = {"commit": "a" * 40, "dirty": False}
    source_tree = root / "retained"
    source_tree.mkdir()
    monkeypatch.setattr(interpreter, "RUNTIME_ROOT", source_tree)
    for module in (prep, inputs, operator):
        monkeypatch.setattr(module, "_harness_revision", lambda: dict(runtime))
    dependency = root / "dependency.py"
    dependency.write_text("value = 1\n")
    binary = root / "ollama"
    binary.write_bytes(b"scripted-executable-never-run")
    binary.chmod(0o700)
    models = root / "existing-models"
    models.mkdir()

    def image(source, destination, **kwargs):
        destination.mkdir(mode=0o700)
        _save(destination / "prepared-image.json", b"scripted-image-receipt")

    monkeypatch.setattr(prep, "prepare_image", image)
    # Real captures/staging/runtime/dependency checks; native image tests are separate.
    monkeypatch.setattr(gate, "_prepared_models", lambda path: (path / "models", {}))
    kwargs = {
        "expected_python": Path(sys.executable),
        "expected_commit": runtime["commit"],
        "executables": {"python": Path(sys.executable), "ollama": binary},
        "dependency_files": [dependency],
        "source_models": models,
        "command_env": {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "LC_ALL": "C"},
    }
    return root / "trial", kwargs, source_tree


def test_existing_interpreter_prepares_eight_fresh_cells_without_launch(configured):
    directory, kwargs, source_tree = configured
    manifest = prep.prepare_feedback_candidate(directory, **kwargs)
    assert (
        manifest.runtime_commit == kwargs["expected_commit"] and not manifest.inference_authorized
    )
    assert not Path(manifest.output_directory).exists()
    assert len(list(Path(manifest.staged_directory).glob("cell-*"))) == 8
    assert not list(Path(manifest.staged_directory).glob("*/admitted.json"))
    record = json.loads((directory / "preparation.json").read_bytes())
    assert record["format"] == "offline-public-feedback-preparation-v1"
    assert not record["parent_approval_issued"] and not record["inference_ran"]
    from evals.public_feedback import validate_feedback_plan, visible_check_bytes

    for cell in validate_feedback_plan().cells:
        workspace = Path(manifest.staged_directory) / f"cell-{cell.ordinal:02d}/workspace"
        assert (workspace / "test_visible.py").read_bytes() == visible_check_bytes(cell.arm)
    assert not (directory / "parent-approval.json").exists()
    assert not (source_tree / ".venv").exists()
    with pytest.raises(FileExistsError):
        prep.prepare_feedback_candidate(directory, **kwargs)


@pytest.mark.parametrize(
    "fault",
    [
        "python_alias",
        "local_prefix",
        "local_environment",
        "dirty",
        "environment",
        "dependency",
        "missing_ollama",
        "python_selection",
    ],
)
def test_invalid_selection_refuses_before_any_trial_output(configured, monkeypatch, fault):
    directory, kwargs, source_tree = configured
    if fault == "python_alias":
        alias = directory.parent / "other-python"
        alias.symlink_to(sys.executable)
        assert alias.resolve() == Path(sys.executable).resolve()
        kwargs["expected_python"] = alias
    elif fault == "local_prefix":
        monkeypatch.setattr(interpreter.sys, "prefix", str(source_tree / ".venv"))
    elif fault == "local_environment":
        (source_tree / ".venv").mkdir()
    elif fault == "dirty":
        monkeypatch.setattr(prep, "_harness_revision", lambda: {"commit": "a" * 40, "dirty": True})
    elif fault == "environment":
        kwargs["command_env"] = {"API_KEY": "secret-sentinel", "PATH": "/usr/bin"}
    elif fault == "missing_ollama":
        kwargs["executables"].pop("ollama")
    elif fault == "python_selection":
        kwargs["executables"]["python"] = directory.parent / "other-python"
    else:
        kwargs["dependency_files"] = [directory.parent / "missing.py"]
    with pytest.raises((ValueError, OSError)) as caught:
        prep.prepare_feedback_candidate(directory, **kwargs)
    assert "secret-sentinel" not in str(caught.value)
    assert not directory.exists()


def test_copy_failure_withdraws_and_retains_partial_without_retry(configured, monkeypatch):
    directory, kwargs, _source_tree = configured

    def fail(source, destination, **options):
        destination.mkdir(mode=0o700)
        (destination / "partial").write_bytes(b"preserve")
        raise OSError("scripted clone failure")

    monkeypatch.setattr(prep, "prepare_image", fail)
    with pytest.raises(OSError, match="clone"):
        prep.prepare_feedback_candidate(directory, **kwargs)
    assert (directory / "image/partial").read_bytes() == b"preserve"
    assert (directory / "WITHDRAWN.json").is_file()
    assert (directory / "preparation-failure.json").is_file()
    assert not (directory / "manifest-candidate.json").exists()
    with pytest.raises(FileExistsError):
        prep.prepare_feedback_candidate(directory, **kwargs)


async def test_active_loop_refuses_before_source_observation(configured, monkeypatch):
    directory, kwargs, _source_tree = configured
    monkeypatch.setattr(prep, "_harness_revision", lambda: pytest.fail("source observed"))
    with pytest.raises(ValueError, match="outside an active"):
        prep.prepare_feedback_candidate(directory, **kwargs)
    assert not directory.exists()


def test_separate_candidate_and_rehearsal_paths_keep_independent_fixtures(configured):
    directory, kwargs, _source_tree = configured
    candidate = prep.prepare_feedback_candidate(directory, **kwargs)
    rehearsal = prep.prepare_feedback_candidate(directory.with_name("rehearsal"), **kwargs)
    assert candidate.staged_directory != rehearsal.staged_directory
    for manifest in (candidate, rehearsal):
        assert not Path(manifest.output_directory).exists()
        assert len(list(Path(manifest.staged_directory).glob("cell-*/prepared.json"))) == 8
        assert not list(Path(manifest.staged_directory).glob("cell-*/admitted.json"))
    assert candidate.capture_index_sha256 != rehearsal.capture_index_sha256


def test_late_preparation_receipt_failure_withdraws_preserved_candidate(configured, monkeypatch):
    directory, kwargs, _source_tree = configured
    original = prep._save

    def save(path, raw):
        if path.name == "preparation.json":
            raise OSError("receipt unavailable")
        return original(path, raw)

    monkeypatch.setattr(prep, "_save", save)
    with pytest.raises(OSError, match="receipt unavailable"):
        prep.prepare_feedback_candidate(directory, **kwargs)
    assert (directory / "manifest-candidate.json").exists()
    assert (directory / "WITHDRAWN.json").exists()
    assert not (directory / "preparation.json").exists()
    assert not (directory / "run").exists()
