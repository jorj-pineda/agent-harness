import pytest

from evals.real_run import acceptance_check, load_tasks
from workspace import disposable_workspace


@pytest.mark.parametrize(
    "task_id,module,old,new",
    [
        ("slugify", "slug.py", "[a-z0-9]+", "[a-z]+"),
        ("slugify", "slug.py", "[a-z0-9]+", r"[^\W_]+"),
        ("parse_flags", "flags.py", ".strip()", '.strip(" ")'),
        ("parse_flags", "flags.py", 'startswith("#")', 'startswith("# ")'),
        ("decimal_total", "money.py", "ROUND_HALF_UP", "ROUND_DOWN"),
        ("stable_dedupe", "dedupe.py", "return result", "return sorted(result)"),
        ("stable_dedupe", "dedupe.py", "return result", "return list(set(items))"),
    ],
    ids=[
        "slug-drops-digits",
        "slug-keeps-non-ascii",
        "flags-space-only-trimming",
        "flags-space-after-comment-marker",
        "decimal-truncates",
        "dedupe-sorts",
        "dedupe-unordered",
    ],
)
async def test_original_python_criteria_reject_contract_mutants(
    task_id: str, module: str, old: str, new: str
) -> None:
    task = next(task for task in load_tasks() if task.id == task_id)
    original = (task.fixture / module).read_bytes()
    source = (task.reference / module).read_text()
    assert old in source
    with disposable_workspace(task.fixture) as copy:
        (copy.workspace.root / module).write_text(source.replace(old, new))
        result = await acceptance_check(task, copy.workspace.root)
    assert result.exit_code == 1, result.stdout + result.stderr
    assert (task.fixture / module).read_bytes() == original
