import json
import os
from unittest.mock import patch

import pytest

from harness.check_suggestions import MANIFEST_LIMIT_BYTES, suggest_project_checks
from harness.config import Settings


def test_static_hints_deduplicate_pytest_and_never_evaluate_scripts(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\naddopts = --strict-markers\n")
    (tmp_path / "pyproject.toml").write_text('[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"test:run": "touch private-marker"}, "token": "private-token"})
    )
    with patch("subprocess.run") as run, patch("subprocess.Popen") as popen:
        report = suggest_project_checks(str(tmp_path))
        run.assert_not_called()
        popen.assert_not_called()
    assert report.status == "inspected"
    assert [item.argv for item in report.suggestions] == [
        ["python", "-m", "pytest"],
        ["npm", "run", "test:run"],
    ]
    assert report.suggestions[0].evidence == ["pytest.ini", "pyproject.toml"]
    for item in report.suggestions:
        assert (
            Settings(_env_file=None, project_check_argv=item.argv).project_check_argv == item.argv
        )
    assert "private" not in report.model_dump_json()
    assert not report.commands_executed and not report.configuration_changed
    assert not (tmp_path / "private-marker").exists()


def test_no_guess_from_language_files_dependencies_or_nested_packages(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="demo"\ndependencies=["pytest"]\n')
    (tmp_path / "package.json").write_text('{"scripts":{"test":"vitest"}}')
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "pytest.ini").write_text("[pytest]\n")
    report = suggest_project_checks(str(tmp_path))
    assert report.status == "inspected" and report.suggestions == [] and report.issues == []
    assert suggest_project_checks(None).status == "no_workspace"
    assert suggest_project_checks(str(tmp_path / "missing")).status == "unavailable"


@pytest.mark.parametrize(
    "name,text",
    [
        ("package.json", "{private-marker"),
        ("pyproject.toml", "[private-marker"),
        ("pytest.ini", "[private-marker"),
    ],
)
def test_invalid_manifests_report_codes_without_contents(name, text, tmp_path):
    (tmp_path / name).write_text(text)
    report = suggest_project_checks(str(tmp_path))
    assert report.status == "partial"
    assert report.issues[0].path == name and report.issues[0].code == "invalid_manifest"
    assert "private-marker" not in report.model_dump_json()


def test_exact_read_boundary_and_overflow(tmp_path):
    path = tmp_path / "pytest.ini"
    prefix = "[pytest]\n#"
    path.write_text(prefix + "x" * (MANIFEST_LIMIT_BYTES - len(prefix)))
    assert suggest_project_checks(str(tmp_path)).suggestions[0].argv == ["python", "-m", "pytest"]
    with path.open("a") as target:
        target.write("x")
    result = suggest_project_checks(str(tmp_path))
    assert result.status == "partial" and result.issues[0].code == "too_large"
    assert result.suggestions == []


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "encoding"])
def test_special_or_invalid_files_are_not_silently_accepted(kind, tmp_path):
    path = tmp_path / "pytest.ini"
    if kind == "symlink":
        outside = tmp_path.parent / "outside.ini"
        outside.write_text("[pytest]\nprivate-token=yes\n")
        path.symlink_to(outside)
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    else:
        path.write_bytes(b"[pytest]\n\xff")
    report = suggest_project_checks(str(tmp_path))
    assert report.status == "partial" and report.suggestions == []
    assert report.issues[0].code == "unreadable_or_unsafe"
    assert "private-token" not in report.model_dump_json()


def test_replaced_open_file_is_detected(tmp_path, monkeypatch):
    path = tmp_path / "pytest.ini"
    path.write_text("[pytest]\n")
    original = os.open

    def replace_after_open(name, flags, **kwargs):
        fd = original(name, flags, **kwargs)
        if name == "pytest.ini":
            path.unlink()
            path.write_text("[pytest]\n# replaced\n")
        return fd

    monkeypatch.setattr("harness.check_suggestions.os.open", replace_after_open)
    # supports_dir_fd contains the real open, so preserve the capability predicate.
    monkeypatch.setattr("harness.check_suggestions.os.supports_dir_fd", {replace_after_open})
    result = suggest_project_checks(str(tmp_path))
    assert result.status == "partial" and result.suggestions == []
    assert result.issues[0].code == "changed"
