"""Load embedded support docs into stable, citable chunks.

Each markdown file under `data/corpus/` carries YAML frontmatter (`title`,
`category`) and is split on H2 headings into one chunk per section. Chunk IDs
are derived from the filename stem and a slug of the section heading, so they
stay stable across re-embeds — rewording a section heading renames its chunk,
which is the intended invalidation signal for the downstream vector index.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

CORPUS_DIR = Path(__file__).parent / "corpus"

_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class Chunk(BaseModel):
    """One citable unit of the support corpus."""

    chunk_id: str = Field(..., description="Stable ID used as the citation anchor")
    doc_id: str = Field(..., description="Source filename stem")
    title: str = Field(..., description="Human-readable doc title from frontmatter")
    section: str = Field(..., description="H2 heading this chunk came from")
    category: str = Field(..., description="Doc-level category, e.g. 'returns'")
    source_path: str = Field(..., description="Filename of the source markdown")
    text: str = Field(..., min_length=1, description="Chunk body text")


def _slug(s: str) -> str:
    return _SLUG_RE.sub("-", s.lower()).strip("-")


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        raise ValueError("Document is missing YAML frontmatter (--- title/category block)")
    meta = yaml.safe_load(m.group(1)) or {}
    if not isinstance(meta, dict):
        raise ValueError("Frontmatter must parse to a YAML mapping")
    return meta, raw[m.end() :]


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Return [(section_heading, section_body), ...] split on H2 headings.

    Falls back to a single "Overview" section if the doc has no H2 headings.
    Content before the first H2 is discarded on purpose — it is expected to be
    the H1 title, which we already pull from frontmatter.
    """
    matches = list(_H2_RE.finditer(body))
    if not matches:
        stripped = body.strip()
        return [("Overview", stripped)] if stripped else []
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append((m.group(1).strip(), body[start:end].strip()))
    return sections


def load_chunks(corpus_dir: Path | str = CORPUS_DIR) -> list[Chunk]:
    """Scan `corpus_dir` for *.md files and return every chunk in stable order.

    Order is deterministic: files sorted by name, then sections in appearance
    order within each file. Stable ordering matters for reproducible embeddings
    and for eval snapshots.
    """
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")

    chunks: list[Chunk] = []
    for md_path in sorted(root.glob("*.md")):
        raw = md_path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        doc_id = md_path.stem
        title = str(meta.get("title") or doc_id.replace("_", " ").title())
        category = str(meta.get("category", "general"))
        for heading, text in _split_sections(body):
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{_slug(heading)}",
                    doc_id=doc_id,
                    title=title,
                    section=heading,
                    category=category,
                    source_path=md_path.name,
                    text=text,
                )
            )
    return chunks
