"""Document loaders for .md, .txt and .pdf sources.

WHAT: turn files on disk into `Document` objects (text, source path, page,
metadata). Markdown files carry YAML front matter parsed BY HAND (no PyYAML
dependency): title, source and allowed_roles. PDFs are read with pypdf (lazy
import). Unreadable or unsupported files are skipped with a logged warning —
a corpus build must never crash on one bad file.
WHY: ingestion is the first stage of RAG; loading front matter here means
role-based access control travels with the document from the very start.
MATHS: none.
FAILURE MODE: front matter that omits allowed_roles would silently become
public; the loader defaults to ["all"] which is the conservative choice for
this synthetic corpus. A malformed front-matter block (no closing ---) is
treated as regular text, not an error.
INTERVIEW ONE-LINER: "Access control metadata is attached at ingestion time,
so the ACL cannot be forgotten later in the pipeline."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt", ".pdf")


@dataclass
class Document:
    """One page-level unit of raw corpus text plus its metadata."""

    text: str
    source: str                      # file path or URL-like identifier
    page: int                        # 1-based page number (markdown: 1)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", Path(self.source).stem))

    @property
    def allowed_roles(self) -> list[str]:
        roles = self.metadata.get("allowed_roles", ["all"])
        return list(roles) if isinstance(roles, (list, tuple)) else [str(roles)]


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split '---\\nkey: value\\n---\\nbody' into (metadata, body).

    Hand-rolled on purpose. Supported value syntax:
      key: plain string
      key: comma, separated, list
      key: [bracketed, list]
    Roles are lowercased. The closing '---' must appear within 100 lines or
    the block is treated as ordinary text.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    close = None
    for i in range(1, min(len(lines), 101)):
        if lines[i].strip() == "---":
            close = i
            break
    if close is None:
        return {}, text
    meta: dict[str, Any] = {}
    for line in lines[1:close]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            parts = [p.strip() for p in value[1:-1].split(",") if p.strip()]
        else:
            parts = [p.strip() for p in value.split(",") if p.strip()]
        if key == "allowed_roles":
            meta[key] = [p.lower() for p in parts]
        elif len(parts) == 1:
            meta[key] = parts[0]
        else:
            meta[key] = parts
    body = "\n".join(lines[close + 1:]).lstrip("\n")
    return meta, body


def load_markdown(path: Path) -> list[Document]:
    meta, body = parse_front_matter(Path(path).read_text(encoding="utf-8"))
    metadata: dict[str, Any] = dict(meta)
    metadata.setdefault("title", Path(path).stem)
    metadata.setdefault("source", Path(path).name)
    # Missing allowed_roles defaults to ["all"]: a policy with no role list is
    # treated as public in this synthetic corpus.
    metadata.setdefault("allowed_roles", ["all"])
    return [Document(text=body, source=str(path), page=1, metadata=metadata)]


def load_text(path: Path) -> list[Document]:
    text = Path(path).read_text(encoding="utf-8")
    return [Document(text=text, source=str(path), page=1,
                     metadata={"title": Path(path).stem, "source": Path(path).name,
                               "allowed_roles": ["all"]})]


def load_pdf(path: Path) -> list[Document]:
    """One Document per page. pypdf is imported lazily so the rest of the
    repository (and the test suite) runs without it installed."""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - exercised via monkeypatch test
        log.warning("pypdf is not installed; skipping PDF %s", path)
        return []
    reader = PdfReader(str(path))
    out: list[Document] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            out.append(Document(text=text, source=str(path), page=i + 1,
                                metadata={"title": Path(path).stem,
                                          "source": Path(path).name,
                                          "allowed_roles": ["all"]}))
    return out


def load_file(path: str | Path) -> list[Document]:
    """Load one file, dispatching on suffix. Never raises: unreadable or
    unsupported files are logged and skipped."""
    p = Path(path)
    try:
        if not p.is_file():
            log.warning("Skipping missing file %s", p)
            return []
        if p.suffix.lower() in (".md", ".markdown"):
            return load_markdown(p)
        if p.suffix.lower() == ".txt":
            return load_text(p)
        if p.suffix.lower() == ".pdf":
            return load_pdf(p)
        log.warning("Unsupported suffix %s for %s; skipped", p.suffix, p)
        return []
    except Exception as exc:  # noqa: BLE001 - ingestion must not crash
        log.warning("Could not read %s (%s: %s); skipped", p, type(exc).__name__, exc)
        return []


def load_directory(directory: str | Path) -> list[Document]:
    """Load every supported file under `directory`, sorted by path so builds
    are deterministic. Hidden files are ignored."""
    root = Path(directory)
    if not root.is_dir():
        log.warning("Data directory %s does not exist", root)
        return []
    paths = sorted(
        p for p in root.rglob("*")
        if p.is_file() and not p.name.startswith(".")
        and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    docs: list[Document] = []
    for p in paths:
        docs.extend(load_file(p))
    log.info("Loaded %d documents from %d files in %s", len(docs), len(paths), root)
    return docs
