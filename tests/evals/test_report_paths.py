import hashlib
from unittest.mock import AsyncMock

import pytest

from evals import real_run
from harness.config import Settings
from tests.api.conftest import ScriptedProvider, make_response
from tests.evals.test_task_catalogs import catalog


def tree_bytes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize(
    "target", ["manifest", "fixture", "acceptance", "reference", "new-fixture"]
)
def test_overlapping_report_refuses_before_controls_and_preserves_inputs(
    monkeypatch, tmp_path, target
):
    manifest, _ = catalog(tmp_path, partition="development")
    task = real_run.load_tasks(manifest)[0]
    report = {
        "manifest": manifest,
        "fixture": task.fixture / "calc.py",
        "acceptance": task.acceptance / "test_acceptance.py",
        "reference": task.reference / "calc.py",
        "new-fixture": task.fixture / "new-dir" / "report.json",
    }[target]
    before = tree_bytes(tmp_path)
    monkeypatch.setattr(real_run, "validate_task", lambda *args: pytest.fail("controls allocated"))
    monkeypatch.setattr(
        real_run, "build_configured_provider", lambda *args: pytest.fail("provider")
    )
    with pytest.raises(ValueError, match="Report path overlaps"):
        real_run.main(["--task-manifest", str(manifest), "--report", str(report)])
    assert tree_bytes(tmp_path) == before


@pytest.mark.parametrize("alias", ["directory", "destination", "inside-link-outside"])
def test_report_aliases_cannot_bypass_input_protection(monkeypatch, tmp_path, alias):
    manifest, _ = catalog(tmp_path, partition="development")
    task = real_run.load_tasks(manifest)[0]
    outside = tmp_path / "safe"
    outside.mkdir()
    if alias == "directory":
        link = outside / "alias"
        link.symlink_to(task.fixture, target_is_directory=True)
        report = link / "report.json"
    elif alias == "destination":
        report = outside / "report.json"
        report.symlink_to(task.reference / "calc.py")
    else:
        (outside / "result.json").write_text("preserve")
        report = task.fixture / "report.json"
        report.symlink_to(outside / "result.json")
    before = tree_bytes(tmp_path)
    monkeypatch.setattr(real_run, "validate_task", lambda *args: pytest.fail("controls allocated"))
    with pytest.raises(ValueError, match="Report path overlaps"):
        real_run.main(["--task-manifest", str(manifest), "--report", str(report)])
    assert tree_bytes(tmp_path) == before


def test_safe_sibling_report_runs_and_parent_retarget_refuses_before_request(monkeypatch, tmp_path):
    manifest, _ = catalog(tmp_path, partition="development")
    task = real_run.load_tasks(manifest)[0]
    safe = tmp_path / "safe"
    safe.mkdir()
    link = tmp_path / "reports"
    link.symlink_to(safe, target_is_directory=True)
    provider = ScriptedProvider()
    provider.script(make_response(content="No patch"))
    provider.aclose = AsyncMock()
    monkeypatch.setattr(real_run, "get_settings", lambda: Settings(_env_file=None))

    def construct(*args):
        link.unlink()
        link.symlink_to(task.fixture, target_is_directory=True)
        return provider

    monkeypatch.setattr(real_run, "build_configured_provider", construct)
    before = tree_bytes(task.fixture)
    with pytest.raises(ValueError, match="Report path overlaps"):
        real_run.main(["--task-manifest", str(manifest), "--report", str(link / "report.json")])
    assert tree_bytes(task.fixture) == before and provider.calls == []
    provider.aclose.assert_awaited_once()
    assert (safe / "report.json").is_file()


def test_report_beside_custom_manifest_is_allowed(monkeypatch, tmp_path):
    manifest, _ = catalog(tmp_path, partition="development")
    provider = ScriptedProvider()
    provider.script(make_response(content="No patch"))
    provider.aclose = AsyncMock()
    monkeypatch.setattr(real_run, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(real_run, "build_configured_provider", lambda *args: provider)
    report = tmp_path / "result.json"
    assert real_run.main(["--task-manifest", str(manifest), "--report", str(report)]) == 0
    assert report.is_file() and len(provider.calls) == 1
    provider.aclose.assert_awaited_once()
