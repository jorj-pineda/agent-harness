import json
import stat

import pytest

from evals import heldout_workspaces as prepared
from evals.heldout_guard import HeldoutDriftError


@pytest.mark.parametrize(
    "task_id", ["ho_iso_week", "ho_rolling_mean", "ho_endpoint_record", "ho_cache_policy"]
)
def test_prepared_copy_is_verified_at_stable_named_path_without_provider(tmp_path, task_id):
    directory = tmp_path / task_id
    receipt = prepared.prepare_fixture(task_id, directory)
    root = prepared.verify_prepared(directory, task_id=task_id)
    assert root == directory.resolve() / "workspace"
    assert receipt.task_id == task_id
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((directory / "prepared.json").stat().st_mode) == 0o600
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(FileExistsError):
        prepared.prepare_fixture(task_id, directory)
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


@pytest.mark.parametrize(
    "defect",
    [
        "bytes",
        "mode",
        "extra",
        "receipt-task",
        "receipt-seal",
        "receipt-mode",
        "receipt-link",
        "workspace-link",
        "outer-extra",
    ],
)
def test_unknown_or_changed_prepared_state_refuses_without_reset(tmp_path, defect):
    directory = tmp_path / "cell-01"
    prepared.prepare_fixture("ho_iso_week", directory)
    root = directory / "workspace"
    path = root / "week_key.py"
    receipt = directory / "prepared.json"
    if defect == "bytes":
        path.write_text("# changed\n")
    elif defect == "mode":
        path.chmod(0o600)
    elif defect == "extra":
        (root / "unknown.py").write_text("unknown")
    elif defect.startswith("receipt-") and defect in ("receipt-task", "receipt-seal"):
        data = json.loads(receipt.read_text())
        data["task_id" if defect == "receipt-task" else "seal_sha256"] = (
            "ho_cache_policy" if defect == "receipt-task" else "0" * 64
        )
        receipt.write_text(json.dumps(data))
    elif defect == "receipt-mode":
        receipt.chmod(0o644)
    elif defect == "receipt-link":
        moved = tmp_path / "receipt.json"
        receipt.rename(moved)
        receipt.symlink_to(moved)
    elif defect == "workspace-link":
        moved = tmp_path / "moved"
        root.rename(moved)
        root.symlink_to(moved, target_is_directory=True)
    else:
        (directory / "unknown").write_text("unknown")
    with pytest.raises((ValueError, OSError)):
        prepared.verify_prepared(directory, task_id="ho_iso_week")
    assert directory.exists()  # No cleanup/reset on refusal.


def test_preparation_never_uses_or_resets_existing_user_directory(tmp_path):
    directory = tmp_path / "owned-by-user"
    directory.mkdir()
    sentinel = directory / "work.txt"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        prepared.prepare_fixture("ho_iso_week", directory)
    assert sentinel.read_text() == "preserve"


def test_git_workspace_and_unknown_task_refuse_before_destination_creation(tmp_path):
    repo = tmp_path / "user-repo"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: /external/worktree")
    with pytest.raises(ValueError, match="Git workspaces"):
        prepared.prepare_fixture("ho_iso_week", repo / "cell")
    assert not (repo / "cell").exists()
    with pytest.raises(ValueError, match="Unknown"):
        prepared.prepare_fixture("unknown", tmp_path / "cell")
    assert not (tmp_path / "cell").exists()


def test_wrong_expected_task_cannot_rebind_existing_copy(tmp_path):
    directory = tmp_path / "cell"
    prepared.prepare_fixture("ho_iso_week", directory)
    with pytest.raises(HeldoutDriftError):
        prepared.verify_prepared(directory, task_id="ho_cache_policy")
