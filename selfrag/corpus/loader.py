"""Read a corpus directory into Documents.

.txt and .md are read directly; .pdf/.docx/.html/.csv/.xlsx need optional parsers and are skipped
with a warning when theirs is missing. parse_file is shared with web ingestion.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Optional


@dataclass
class Document:
    doc_id: str      # path relative to the corpus root
    path: str
    text: str


_BRACE = re.compile(r"\{([^}]*)\}")


def _extensions_from_glob(glob: str) -> list[str]:
    # pathlib globs don't expand braces, so pull the extensions out ourselves
    m = _BRACE.search(glob)
    if m:
        return [f".{e.strip().lstrip('.')}" for e in m.group(1).split(",") if e.strip()]
    suffix = Path(glob).suffix
    return [suffix] if suffix else [".txt", ".md"]


def _read_plain(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> Optional[str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> Optional[str]:
    try:
        import docx
    except ImportError:
        return None
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs]
    for table in d.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts)


class _HTMLText(HTMLParser):
    _SKIP = {"script", "style", "noscript", "template"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
              "section", "article", "header", "footer", "blockquote", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self._chunks.append("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self._chunks.append(data)

    def text(self) -> str:
        raw = "".join(self._chunks)
        lines = [ln.strip() for ln in raw.splitlines()]
        return "\n".join(ln for ln in lines if ln)


def _read_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        import trafilatura
        extracted = trafilatura.extract(raw)
        if extracted:
            return extracted
    except ImportError:
        pass
    p = _HTMLText()
    p.feed(raw)
    return p.text()


def _rows_to_lines(df) -> list[str]:
    lines = []
    for _, row in df.iterrows():
        cells = [f"{col}: {val}" for col, val in row.items()
                 if val is not None and str(val) not in ("", "nan", "NaT")]
        if cells:
            lines.append(" | ".join(cells))
    return lines


def _read_csv(path: Path) -> Optional[str]:
    try:
        import pandas as pd
    except ImportError:
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return "\n".join(_rows_to_lines(df))


def _read_xlsx(path: Path) -> Optional[str]:
    try:
        import pandas as pd
    except ImportError:
        return None
    try:
        sheets = pd.read_excel(path, sheet_name=None, dtype=str)
    except ImportError:   # openpyxl missing
        return None
    parts = []
    for name, df in sheets.items():
        lines = _rows_to_lines(df.fillna(""))
        if lines:
            parts.append(f"# sheet: {name}\n" + "\n".join(lines))
    return "\n\n".join(parts)


PARSERS: dict[str, Callable[[Path], Optional[str]]] = {
    ".txt": _read_plain,
    ".md": _read_plain,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".html": _read_html,
    ".htm": _read_html,
    ".csv": _read_csv,
    ".xlsx": _read_xlsx,
}


def parse_file(path: Path) -> Optional[str]:
    """None if the extension is unsupported or its parser isn't installed."""
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        return None
    return parser(path)


def document_title(doc: Document) -> str:
    first = doc.text.lstrip().split("\n", 1)[0].strip()
    return first.lstrip("#").strip() or Path(doc.doc_id).stem


def load_corpus(corpus_path: str | Path, corpus_glob: str = "**/*.{txt,md}") -> list[Document]:
    root = Path(corpus_path)
    if not root.exists():
        raise FileNotFoundError(f"corpus_path does not exist: {root}")

    exts = {e.lower() for e in _extensions_from_glob(corpus_glob)}
    docs: list[Document] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        text = parse_file(p)
        if text is None:
            warnings.warn(f"skipping {p}: no parser for {p.suffix} (missing optional dependency?)",
                          stacklevel=2)
            continue
        text = text.strip()
        if not text:
            continue
        docs.append(Document(doc_id=str(p.relative_to(root)), path=str(p), text=text))

    if not docs:
        raise ValueError(f"no documents found under {root} matching {sorted(exts)}")
    return docs
