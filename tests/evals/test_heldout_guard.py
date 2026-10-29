import hashlib
import json
import shutil
from contextlib import contextmanager
from dataclasses import replace

import pytest

from evals import heldout_guard, real_run
from evals.heldout_guard import HeldoutDriftError, HeldoutRegistration, verify_catalog
from harness.config import Settings
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response


@pytest.fixture
def registered(tmp_path, monkeypatch):
    original = heldout_guard.REGISTRATIONS[0]
    root = tmp_path / "v1"
    shutil.copytree(original.root, root)
    registration = HeldoutRegistration(root, original.seal_sha256)
    monkeypatch.setattr(heldout_guard, "REGISTRATIONS", (registration,))
    return registration


def test_reviewed_catalog_and_pin_index_match(registered):
    seal = verify_catalog(registered)
    assert len(seal.files) == 18
    assert (
        hashlib.sha256((registered.root / "seal.json").read_bytes()).hexdigest()
        == registered.seal_sha256
    )


@pytest.mark.parametrize(
    "defect",
    [
        "changed",
        "mode",
        "missing",
        "extra",
        "empty-dir",
        "symlink",
        "seal-missing",
        "seal-changed",
        "seal-mode",
        "oversized-seal",
        "header-development",
    ],
)
def test_catalog_drift_refuses_before_workspace_or_provider_allocation(
    registered, monkeypatch, defect
):
    root = registered.root
    path = root / "fixtures/ho_iso_week/week_key.py"
    if defect == "changed":
        path.write_text(path.read_text() + "# drift\n")
    elif defect == "mode":
        path.chmod(0o600)
    elif defect == "missing":
        path.unlink()
    elif defect == "extra":
        (root / "unknown.txt").write_text("unknown")
    elif defect == "empty-dir":
        (root / "unknown-dir").mkdir()
    elif defect == "symlink":
        outside = root.parent / "outside.py"
        outside.write_text("outside-placeholder")
        path.unlink()
        path.symlink_to(outside)
    elif defect == "seal-missing":
        (root / "seal.json").unlink()
    elif defect == "seal-changed":
        (root / "seal.json").write_text((root / "seal.json").read_text() + " ")
    elif defect == "seal-mode":
        (root / "seal.json").chmod(0o600)
    elif defect == "oversized-seal":
        (root / "seal.json").write_bytes(b"x" * 64_001)
    else:
        manifest = root / "tasks.yaml"
        manifest.write_text(
            manifest.read_text().replace("partition: held_out", "partition: development")
        )

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid catalog allocated evaluation resources")

    monkeypatch.setattr(real_run, "disposable_workspace", forbidden)
    monkeypatch.setattr(real_run, "build_configured_provider", forbidden)
    with pytest.raises(HeldoutDriftError):
        real_run.main(["--task-manifest", str(root / "tasks.yaml")])


def test_unknown_environment_file_is_not_read(registered, monkeypatch):
    (registered.root / ".env").write_text("placeholder-only\n")
    reads = []
    original = heldout_guard._read_regular

    def observed(path, limit):
        reads.append(path)
        return original(path, limit)

    monkeypatch.setattr(heldout_guard, "_read_regular", observed)
    with pytest.raises(HeldoutDriftError):
        verify_catalog(registered)
    assert reads == [registered.root / "seal.json"]


def test_modified_data_cannot_be_rebound_with_modified_index(registered):
    index = registered.root / "seal.json"
    pins = json.loads(index.read_text())
    name = "reference/ho_iso_week/week_key.py"
    path = registered.root / name
    path.write_text(path.read_text() + "# changed\n")
    raw = path.read_bytes()
    pins["files"][name].update(sha256=hashlib.sha256(raw).hexdigest(), size=len(raw))
    index.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n")
    with pytest.raises(HeldoutDriftError):
        verify_catalog(registered)


@pytest.mark.parametrize("field", ["request", "category", "task_set", "relabelled-source"])
async def test_spoofed_task_metadata_refuses_before_copy(registered, monkeypatch, field):
    task = real_run.load_tasks(registered.root / "tasks.yaml")[0]
    if field == "relabelled-source":
        task = replace(
            task,
            fixture=real_run.load_tasks()[0].fixture,
            task_set=task.task_set.model_copy(update={"partition": "development"}),
        )
    else:
        value = {"request": "Different task", "category": "repository_question", "task_set": None}[
            field
        ]
        task = replace(task, **{field: value})
    monkeypatch.setattr(
        real_run, "disposable_workspace", lambda *args: pytest.fail("Spoofed task copied")
    )
    with pytest.raises(HeldoutDriftError):
        await real_run.run_task(
            task, provider=ScriptedProvider(), settings=Settings(_env_file=None), mode="harness"
        )


class DriftingProvider(ScriptedProvider):
    def __init__(self, root):
        super().__init__()
        self.root = root

    async def chat(self, messages, **kwargs):
        response = await super().chat(messages, **kwargs)
        if len(self.calls) == 1:
            path = self.root / "reference/ho_iso_week/week_key.py"
            path.write_text(path.read_text() + "# drift during request\n")
        return response


def test_root_symlink_refuses_before_index_read(registered, monkeypatch):
    target = registered.root.parent / "moved"
    registered.root.rename(target)
    registered.root.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(
        heldout_guard, "_read_regular", lambda *args: pytest.fail("Followed root link")
    )
    with pytest.raises(HeldoutDriftError):
        verify_catalog(registered)


async def test_corrupt_disposable_copy_refuses_before_model_work(registered, monkeypatch):
    task = real_run.load_tasks(registered.root / "tasks.yaml")[0]
    original = real_run.disposable_workspace

    @contextmanager
    def corrupted(source):
        with original(source) as copy:
            (copy.workspace.root / "week_key.py").write_text("# corrupt copy\n")
            yield copy

    monkeypatch.setattr(real_run, "disposable_workspace", corrupted)
    provider = ScriptedProvider()
    with pytest.raises(HeldoutDriftError):
        await real_run.run_task(
            task, provider=provider, settings=Settings(_env_file=None), mode="harness"
        )
    assert provider.calls == []


async def test_unregistered_heldout_cannot_execute(registered, monkeypatch):
    task = real_run.load_tasks(registered.root / "tasks.yaml")[0]
    monkeypatch.setattr(heldout_guard, "REGISTRATIONS", ())
    monkeypatch.setattr(
        real_run, "disposable_workspace", lambda *args: pytest.fail("Unregistered task copied")
    )
    with pytest.raises(HeldoutDriftError, match="Unregistered"):
        await real_run.run_task(
            task, provider=ScriptedProvider(), settings=Settings(_env_file=None), mode="harness"
        )


def test_drift_after_validation_refuses_before_provider_construction(registered, monkeypatch):
    def settings():
        path = registered.root / "reference/ho_iso_week/week_key.py"
        path.write_text(path.read_text() + "# drift after validation\n")
        return Settings(_env_file=None)

    monkeypatch.setattr(real_run, "get_settings", settings)
    monkeypatch.setattr(
        real_run, "build_configured_provider", lambda *args: pytest.fail("Constructed after drift")
    )
    with pytest.raises(HeldoutDriftError):
        real_run.main(
            ["--task-manifest", str(registered.root / "tasks.yaml"), "--task", "ho_iso_week"]
        )


async def test_drift_after_grading_invalidates_even_a_passing_artifact(registered, monkeypatch):
    task = real_run.load_tasks(registered.root / "tasks.yaml")[0]
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[
                ToolCall(
                    id="edit",
                    name="write_file",
                    arguments={
                        "path": "week_key.py",
                        "content": (task.reference / "week_key.py").read_text(),
                    },
                )
            ]
        ),
        make_response(content="Native completion"),
    )
    original = real_run.run_process
    graded = []

    async def drifting_grade(*args, **kwargs):
        result = await original(*args, **kwargs)
        graded.append(result.exit_code)
        path = registered.root / "reference/ho_iso_week/week_key.py"
        path.write_text(path.read_text() + "# drift after grade\n")
        return result

    monkeypatch.setattr(real_run, "run_process", drifting_grade)
    result = await real_run.run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert graded == [0]
    assert result.completion_status == "completed" and not result.passed
    assert result.termination == "error:HeldoutDriftError"


async def test_midrun_drift_prevents_next_request_and_acceptance(registered):
    task = real_run.load_tasks(registered.root / "tasks.yaml")[0]
    provider = DriftingProvider(registered.root)
    provider.script(
        make_response(
            tool_calls=[ToolCall(id="read", name="read_file", arguments={"path": "week_key.py"})],
            prompt_tokens=10,
            completion_tokens=5,
        ),
        make_response(content="Must never dispatch"),
    )
    result = await real_run.run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert len(provider.calls) == 1
    assert result.termination == "error:HeldoutDriftError" and not result.passed
    assert "not certified" in result.acceptance_output
    assert result.prompt_tokens == 10 and result.completion_tokens == 5


def test_cli_retains_drift_row_closes_provider_and_stops_cohort(registered, monkeypatch, tmp_path):
    provider = DriftingProvider(registered.root)
    provider.script(
        make_response(
            tool_calls=[ToolCall(id="read", name="read_file", arguments={"path": "week_key.py"})]
        )
    )
    closed = []

    async def close():
        closed.append(True)

    provider.aclose = close
    monkeypatch.setattr(real_run, "build_configured_provider", lambda *args: provider)
    monkeypatch.setattr(real_run, "get_settings", lambda: Settings(_env_file=None))
    report = tmp_path / "report.json"
    with pytest.raises(HeldoutDriftError):
        real_run.main(
            [
                "--task-manifest",
                str(registered.root / "tasks.yaml"),
                "--task",
                "ho_iso_week",
                "--mode",
                "both",
                "--repeats",
                "2",
                "--report",
                str(report),
            ]
        )
    rows = json.loads(report.read_text())["results"]
    assert len(rows) == 1 and rows[0]["termination"] == "error:HeldoutDriftError"
    assert len(provider.calls) == 1 and closed == [True]


async def test_post_response_drift_never_certifies_a_grade(registered, monkeypatch):
    task = real_run.load_tasks(registered.root / "tasks.yaml")[0]
    provider = DriftingProvider(registered.root)
    provider.script(make_response(content="Native completion"))
    monkeypatch.setattr(
        real_run, "acceptance_check", lambda *args, **kwargs: pytest.fail("Grading changed data")
    )
    result = await real_run.run_task(
        task, provider=provider, settings=Settings(_env_file=None), mode="harness"
    )
    assert result.completion_status == "completed" and not result.passed
    assert result.termination == "error:HeldoutDriftError"
    assert result.acceptance_identity["status"] == "unavailable"
