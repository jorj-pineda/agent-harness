from __future__ import annotations

from pathlib import Path

import pytest

from data.corpus import CORPUS_DIR, Chunk, load_chunks


def test_real_corpus_loads_and_is_nonempty() -> None:
    chunks = load_chunks()
    assert len(chunks) >= 20, "Corpus should have enough chunks for retrieval variety"


def test_real_corpus_covers_multiple_categories() -> None:
    chunks = load_chunks()
    categories = {c.category for c in chunks}
    # Diversity matters for RAG scoring and eval scenarios.
    assert len(categories) >= 4, f"Only {len(categories)} categories: {categories}"


def test_real_corpus_chunk_ids_are_unique() -> None:
    chunks = load_chunks()
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique — duplicates break citations"


def test_real_corpus_chunks_are_pydantic_validated() -> None:
    chunks = load_chunks()
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.text.strip(), f"Empty text for {c.chunk_id}"
        assert c.section, f"Missing section on {c.chunk_id}"
        assert c.chunk_id.startswith(f"{c.doc_id}#"), (
            f"chunk_id should be doc_id#section-slug, got {c.chunk_id}"
        )


def test_real_corpus_is_loaded_in_stable_order() -> None:
    first = load_chunks()
    second = load_chunks()
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_load_chunks_parses_synthetic_doc(tmp_path: Path) -> None:
    doc = tmp_path / "one.md"
    doc.write_text(
        "---\n"
        "title: One Doc\n"
        "category: test\n"
        "---\n\n"
        "# One Doc\n\n"
        "## First section\n\n"
        "First body.\n\n"
        "## Second section\n\n"
        "Second body.\n",
        encoding="utf-8",
    )
    chunks = load_chunks(tmp_path)
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "one#first-section"
    assert chunks[0].section == "First section"
    assert chunks[0].title == "One Doc"
    assert chunks[0].category == "test"
    assert chunks[0].source_path == "one.md"
    assert chunks[0].text == "First body."
    assert chunks[1].chunk_id == "one#second-section"


def test_load_chunks_sorts_files_by_name(tmp_path: Path) -> None:
    for name in ("zeta.md", "alpha.md", "mid.md"):
        (tmp_path / name).write_text(
            f"---\ntitle: {name}\ncategory: t\n---\n\n## S\n\nbody\n",
            encoding="utf-8",
        )
    chunks = load_chunks(tmp_path)
    assert [c.doc_id for c in chunks] == ["alpha", "mid", "zeta"]


def test_missing_frontmatter_raises(tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_text("# No frontmatter\n\n## Section\n\nbody\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frontmatter"):
        load_chunks(tmp_path)


def test_doc_with_no_h2_falls_back_to_overview(tmp_path: Path) -> None:
    (tmp_path / "flat.md").write_text(
        "---\ntitle: Flat\ncategory: t\n---\n\nJust a paragraph, no H2 headings at all.\n",
        encoding="utf-8",
    )
    chunks = load_chunks(tmp_path)
    assert len(chunks) == 1
    assert chunks[0].section == "Overview"
    assert chunks[0].chunk_id == "flat#overview"
    assert "paragraph" in chunks[0].text


def test_empty_section_is_dropped(tmp_path: Path) -> None:
    (tmp_path / "gaps.md").write_text(
        "---\ntitle: Gaps\ncategory: t\n---\n\n## Empty\n\n## Filled\n\nonly this one has text\n",
        encoding="utf-8",
    )
    chunks = load_chunks(tmp_path)
    assert len(chunks) == 1
    assert chunks[0].section == "Filled"


def test_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path / "does_not_exist")


def test_title_defaults_from_filename_when_missing(tmp_path: Path) -> None:
    (tmp_path / "my_doc.md").write_text(
        "---\ncategory: t\n---\n\n## S\n\nbody\n",
        encoding="utf-8",
    )
    chunks = load_chunks(tmp_path)
    assert chunks[0].title == "My Doc"


def test_corpus_dir_constant_points_at_real_corpus() -> None:
    assert CORPUS_DIR.is_dir()
    assert any(CORPUS_DIR.glob("*.md"))
