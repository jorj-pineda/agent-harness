import json
import shutil

import pytest

import evals.public_workspaces as public


@pytest.mark.parametrize("ordinal", range(1, 7))
def test_each_public_cell_has_verified_named_copy_and_single_use(tmp_path, ordinal):
    directory = tmp_path / f"cell-{ordinal}"
    receipt = public.prepare_public_cell(ordinal, directory)
    root = public.verify_public_cell(directory, ordinal=ordinal)
    assert root == directory / "workspace" and receipt.ordinal == ordinal
    assert directory.stat().st_mode & 0o777 == 0o700
    with public.admit_public_cell(directory, ordinal=ordinal) as copy:
        (copy.workspace.root / "created.py").write_text("value = 1\n")
        assert "created.py" in copy.changes().added
    assert (root / "created.py").exists()
    with (
        pytest.raises(ValueError, match="inventory"),
        public.admit_public_cell(directory, ordinal=ordinal),
    ):
        pytest.fail("admitted twice")


@pytest.mark.parametrize(
    "defect", ["bytes", "extra", "mode", "receipt", "receipt-mode", "link", "outer", "outer-mode"]
)
def test_unknown_changes_refuse_without_reset(tmp_path, defect):
    directory = tmp_path / "cell"
    public.prepare_public_cell(1, directory)
    root = directory / "workspace"
    receipt = directory / "prepared.json"
    if defect == "bytes":
        (root / "calc.py").write_text("changed\n")
    elif defect == "extra":
        (root / "extra.py").write_text("extra\n")
    elif defect == "mode":
        path = root / "calc.py"
        path.chmod(path.stat().st_mode ^ 0o100)
    elif defect == "receipt":
        data = json.loads(receipt.read_bytes())
        data["ordinal"] = 4
        receipt.write_text(json.dumps(data))
    elif defect == "receipt-mode":
        receipt.chmod(0o644)
    elif defect == "link":
        moved = tmp_path / "moved"
        root.rename(moved)
        root.symlink_to(moved, target_is_directory=True)
    elif defect == "outer":
        (directory / "unknown").write_text("preserve")
    else:
        directory.chmod(0o755)
    with pytest.raises((ValueError, OSError)):
        public.verify_public_cell(directory, ordinal=1)
    assert directory.exists() and not (directory / "admitted.json").exists()


def test_occupied_destination_and_wrong_ordinal_refuse(tmp_path):
    directory = tmp_path / "cell"
    public.prepare_public_cell(1, directory)
    with pytest.raises(FileExistsError):
        public.prepare_public_cell(1, directory)
    with pytest.raises(ValueError, match="receipt"):
        public.verify_public_cell(directory, ordinal=4)


@pytest.mark.parametrize("ordinal", [0, 7, True])
def test_invalid_selection_refuses_before_allocation(tmp_path, ordinal):
    directory = tmp_path / "cell"
    with pytest.raises(ValueError):
        public.prepare_public_cell(ordinal, directory)
    assert not directory.exists()


def test_git_workspace_refuses(tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError, match="Git"):
        public.prepare_public_cell(1, tmp_path / "cell")


def test_failed_copy_keeps_partial_evidence_without_receipt(tmp_path, monkeypatch):
    def fail(source, destination, **kwargs):
        destination.mkdir()
        (destination / "partial").write_text("keep")
        raise OSError("copy failure")

    monkeypatch.setattr(shutil, "copytree", fail)
    directory = tmp_path / "cell"
    with pytest.raises(OSError, match="copy failure"):
        public.prepare_public_cell(1, directory)
    assert (directory / "workspace/partial").read_text() == "keep"
    assert not (directory / "prepared.json").exists()


def test_failure_after_marker_consumes_admission(tmp_path, monkeypatch):
    directory = tmp_path / "cell"
    public.prepare_public_cell(1, directory)
    original = public._selection
    calls = 0

    def changed(ordinal):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("post-marker source drift")
        return original(ordinal)

    monkeypatch.setattr(public, "_selection", changed)
    with (
        pytest.raises(ValueError, match="post-marker"),
        public.admit_public_cell(directory, ordinal=1),
    ):
        pytest.fail("yielded after drift")
    assert (directory / "admitted.json").exists()
    monkeypatch.setattr(public, "_selection", original)
    with (
        pytest.raises(ValueError, match="inventory"),
        public.admit_public_cell(directory, ordinal=1),
    ):
        pytest.fail("retried consumed admission")


def test_context_failure_keeps_edits_and_consumed_marker(tmp_path):
    directory = tmp_path / "cell"
    public.prepare_public_cell(1, directory)
    with (
        pytest.raises(RuntimeError, match="caller failure"),
        public.admit_public_cell(directory, ordinal=1) as copy,
    ):
        (copy.workspace.root / "partial.py").write_text("keep\n")
        raise RuntimeError("caller failure")
    assert (directory / "workspace/partial.py").read_text() == "keep\n"
    assert (directory / "admitted.json").exists()


def test_symlink_parent_alias_refuses_before_allocation(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical"):
        public.prepare_public_cell(1, alias / "cell")
    assert not (target / "cell").exists()
