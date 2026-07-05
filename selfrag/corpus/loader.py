"""Read a corpus directory into Documents.

Supports .txt and .md natively; .pdf if a parser is available (skipped with a warning otherwise).
The config glob may use brace syntax (``**/*.{txt,md,pdf}``), which Python's glob doesn't expand,
so we parse the extension set ourselves.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    doc_id: str      # stable id: path relative to corpus root
    path: str
    text: str


_BRACE = re.compile(r"\{([^}]*)\}")


def _extensions_from_glob(glob: str) -> list[str]:
    """Extract the extension set from a glob like ``**/*.{txt,md,pdf}`` -> ['.txt','.md','.pdf'].

    Falls back to the literal suffix if there is no brace group.
    """
    m = _BRACE.search(glob)
    if m:
        return [f".{e.strip().lstrip('.')}" for e in m.group(1).split(",") if e.strip()]
    suffix = Path(glob).suffix
    return [suffix] if suffix else [".txt", ".md"]


def _read_pdf(path: Path) -> str | None:
    try:
        from pypdf import PdfReader  # optional dependency
    except ImportError:
        return None
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_corpus(corpus_path: str | Path, corpus_glob: str = "**/*.{txt,md}") -> list[Document]:
    root = Path(corpus_path)
    if not root.exists():
        raise FileNotFoundError(f"corpus_path does not exist: {root}")

    exts = {e.lower() for e in _extensions_from_glob(corpus_glob)}
    docs: list[Document] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        if p.suffix.lower() == ".pdf":
            text = _read_pdf(p)
            if text is None:
                warnings.warn(f"skipping {p}: install `pypdf` to read PDFs", stacklevel=2)
                continue
        else:
            text = p.read_text(encoding="utf-8", errors="replace")
        text = text.strip()
        if not text:
            continue
        docs.append(Document(doc_id=str(p.relative_to(root)), path=str(p), text=text))

    if not docs:
        raise ValueError(f"no documents found under {root} matching {sorted(exts)}")
    return docs
