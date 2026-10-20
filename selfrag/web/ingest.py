"""Ingestion of uploads, zip/file URLs and HuggingFace datasets into a project corpus.

This is the part that faces the network, so:
  - extension allowlist, magic-byte check, per-file and per-project size caps
  - zip entry count / size / compression-ratio caps, no absolute or '..' paths, symlinks skipped
  - URL fetches only reach public addresses, every redirect hop is re-checked, downloads are capped
Every file ends up as projects/<id>/corpus/<name>.txt.
"""

from __future__ import annotations

import io
import ipaddress
import json
import re
import socket
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

from ..corpus.loader import PARSERS, parse_file

ALLOWED_EXTS = set(PARSERS.keys())

_ZIP_MAX_ENTRIES = 1000
_ZIP_MAX_RATIO = 200
_FETCH_MAX_REDIRECTS = 3
_FETCH_TIMEOUT_S = 30

_HF_SERVER = "https://datasets-server.huggingface.co"
_HF_ROWS_PER_PAGE = 100
_HF_DEFAULT_MAX_ROWS = 300

# text formats have no reliable magic bytes
_MAGIC = {
    ".pdf": (b"%PDF",),
    ".docx": (b"PK",),
    ".xlsx": (b"PK",),
}


class IngestError(ValueError):
    pass


@dataclass
class IngestLimits:
    max_file_bytes: int = 25 * 1024 * 1024
    max_project_bytes: int = 200 * 1024 * 1024

    @classmethod
    def from_cfg(cls, runtime_cfg: Any) -> "IngestLimits":
        return cls(
            max_file_bytes=int(getattr(runtime_cfg, "max_upload_mb", 25)) * 1024 * 1024,
            max_project_bytes=int(getattr(runtime_cfg, "max_project_mb", 200)) * 1024 * 1024,
        )


_NAME_BAD = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str) -> str:
    base = Path(name.replace("\\", "/")).name
    base = _NAME_BAD.sub("_", base).lstrip(".")
    if not base or not Path(base).stem:
        raise IngestError(f"invalid filename: {name!r}")
    if len(base) > 120:
        base = base[-120:]
    return base


def _check_ext(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise IngestError(f"file type {ext or '(none)'} not allowed "
                          f"(allowed: {', '.join(sorted(ALLOWED_EXTS))})")
    return ext


def _check_magic(ext: str, data: bytes) -> None:
    prefixes = _MAGIC.get(ext)
    if prefixes and not any(data[:8].startswith(p) for p in prefixes):
        raise IngestError(f"file content does not match its {ext} extension")


def _corpus_bytes(corpus_dir: Path) -> int:
    return sum(p.stat().st_size for p in corpus_dir.rglob("*") if p.is_file())


def _unique_target(corpus_dir: Path, stem: str) -> Path:
    target = corpus_dir / f"{stem}.txt"
    n = 1
    while target.exists():
        target = corpus_dir / f"{stem}-{n}.txt"
        n += 1
    return target


def ingest_bytes(project_dir: Path, filename: str, data: bytes,
                 limits: IngestLimits, origin: str = "upload") -> dict[str, Any]:
    name = sanitize_filename(filename)
    ext = _check_ext(name)
    if len(data) > limits.max_file_bytes:
        raise IngestError(f"{name}: {len(data)/1e6:.1f} MB exceeds the "
                          f"{limits.max_file_bytes//(1024*1024)} MB per-file cap")
    corpus_dir = project_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    if _corpus_bytes(corpus_dir) + len(data) > limits.max_project_bytes:
        raise IngestError("project corpus size cap reached")
    _check_magic(ext, data)

    # the loader parsers take paths
    with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        text = parse_file(Path(tmp.name))
    if text is None:
        raise IngestError(f"{name}: no parser available for {ext} "
                          "(is the optional dependency installed on the server?)")
    text = text.strip()

    entry = {"source": name, "origin": origin, "bytes": len(data), "chars": len(text)}
    if not text:
        entry["warning"] = "no extractable text, file skipped"
        return entry
    target = _unique_target(corpus_dir, Path(name).stem)
    target.write_text(text, encoding="utf-8")
    entry["stored"] = target.name
    return entry


def iter_zip_safely(data: bytes, limits: IngestLimits) -> Iterator[tuple[str, bytes]]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise IngestError(f"not a valid zip archive: {e}") from e
    with zf:
        infos = zf.infolist()
        if len(infos) > _ZIP_MAX_ENTRIES:
            raise IngestError(f"zip has {len(infos)} entries (max {_ZIP_MAX_ENTRIES})")
        total = 0
        for info in infos:
            if info.is_dir():
                continue
            name = info.filename
            parts = Path(name).parts
            if name.startswith(("/", "\\")) or ".." in parts:
                raise IngestError(f"zip entry escapes the archive root: {name!r}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:   # symlink
                continue
            if Path(name).suffix.lower() not in ALLOWED_EXTS:
                continue
            if info.compress_size and info.file_size / info.compress_size > _ZIP_MAX_RATIO:
                raise IngestError(f"zip entry {name!r} has a suspicious compression ratio")
            if info.file_size > limits.max_file_bytes:
                continue
            total += info.file_size
            if total > limits.max_project_bytes:
                raise IngestError("zip decompresses past the project size cap")
            with zf.open(info) as fh:
                blob = fh.read(limits.max_file_bytes + 1)
            if len(blob) > limits.max_file_bytes:   # header lied about file_size
                raise IngestError(f"zip entry {name!r} is larger than its declared size")
            yield name, blob


def _host_is_public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
                or ip.is_reserved or ip.is_unspecified):
            return False
    return True


def validate_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise IngestError(f"only http(s) URLs are allowed, got {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise IngestError("URL has no hostname")
    if not _host_is_public(host):
        raise IngestError(f"host {host!r} does not resolve to a public address, refusing "
                          "(private/loopback ranges are blocked)")
    return url


def fetch_url(url: str, max_bytes: int) -> bytes:
    # TODO: resolve-then-fetch leaves a DNS rebinding window. Pin the resolved IP before
    # hosting this anywhere public.
    import httpx

    for _ in range(_FETCH_MAX_REDIRECTS + 1):
        validate_public_url(url)
        with httpx.Client(follow_redirects=False, timeout=_FETCH_TIMEOUT_S) as client:
            with client.stream("GET", url) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location")
                    if not location:
                        raise IngestError("redirect without a Location header")
                    url = str(httpx.URL(url).join(location))
                    continue
                resp.raise_for_status()
                buf = io.BytesIO()
                for chunk in resp.iter_bytes():
                    buf.write(chunk)
                    if buf.tell() > max_bytes:
                        raise IngestError(f"download exceeds the {max_bytes//(1024*1024)} MB cap")
                return buf.getvalue()
    raise IngestError("too many redirects")


def ingest_zip_url(project_dir: Path, url: str, limits: IngestLimits) -> list[dict[str, Any]]:
    data = fetch_url(url, limits.max_project_bytes)
    entries = []
    if url.lower().endswith(".zip") or data[:2] == b"PK":
        for name, blob in iter_zip_safely(data, limits):
            try:
                entries.append(ingest_bytes(project_dir, name, blob, limits, origin=f"zip:{url}"))
            except IngestError as e:
                entries.append({"source": name, "origin": f"zip:{url}", "error": str(e)})
    else:
        name = sanitize_filename(Path(urlparse(url).path).name or "download.txt")
        entries.append(ingest_bytes(project_dir, name, data, limits, origin=f"url:{url}"))
    if not entries:
        raise IngestError("archive contained no ingestible files "
                          f"(allowed: {', '.join(sorted(ALLOWED_EXTS))})")
    return entries


_TEXT_COLUMN_HINTS = ("text", "content", "article", "document", "context", "passage", "body",
                      "abstract", "question")


def parse_hf_spec(spec: str) -> tuple[str, Optional[str], Optional[str]]:
    """'hf://squad', 'hf://user/name:config:split', or a huggingface.co/datasets/... URL."""
    s = spec.strip()
    if s.startswith("hf://"):
        s = s[len("hf://"):]
    elif "huggingface.co/datasets/" in s:
        s = s.split("huggingface.co/datasets/", 1)[1].split("?")[0].strip("/")
    parts = s.split(":")
    dataset = parts[0].strip("/")
    if not dataset:
        raise IngestError(f"cannot parse HF dataset spec: {spec!r}")
    config = parts[1] if len(parts) > 1 and parts[1] else None
    split = parts[2] if len(parts) > 2 and parts[2] else None
    return dataset, config, split


def _hf_get_json(path: str, max_bytes: int) -> Any:
    return json.loads(fetch_url(f"{_HF_SERVER}{path}", max_bytes))


def _pick_text_column(rows: list[dict[str, Any]], requested: Optional[str]) -> str:
    if not rows:
        raise IngestError("dataset returned no rows")
    columns = rows[0].keys()
    if requested:
        if requested not in columns:
            raise IngestError(f"column {requested!r} not in dataset columns: {sorted(columns)}")
        return requested
    for hint in _TEXT_COLUMN_HINTS:
        if hint in columns:
            return hint
    # fall back to the string column with the longest average content
    def avg_len(col: str) -> float:
        vals = [r.get(col) for r in rows if isinstance(r.get(col), str)]
        return sum(len(v) for v in vals) / len(vals) if vals else 0.0
    best = max(columns, key=avg_len)
    if avg_len(best) < 20:
        raise IngestError(f"no text-like column found; specify text_column from {sorted(columns)}")
    return best


def ingest_hf(project_dir: Path, spec: str, limits: IngestLimits,
              text_column: Optional[str] = None,
              max_rows: int = _HF_DEFAULT_MAX_ROWS) -> dict[str, Any]:
    # each row becomes one document
    dataset, config, split = parse_hf_spec(spec)
    from urllib.parse import quote
    ds_q = quote(dataset, safe="")

    if config is None or split is None:
        info = _hf_get_json(f"/splits?dataset={ds_q}", limits.max_file_bytes)
        splits = info.get("splits") or []
        if not splits:
            raise IngestError(f"HF dataset {dataset!r} has no queryable splits")
        chosen = next((s for s in splits
                       if (config is None or s["config"] == config)
                       and (split is None or s["split"] == split)), None)
        if chosen is None:
            raise IngestError(f"no split matching config={config} split={split} for {dataset!r}")
        config, split = chosen["config"], chosen["split"]

    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < max_rows:
        length = min(_HF_ROWS_PER_PAGE, max_rows - len(rows))
        page = _hf_get_json(
            f"/rows?dataset={ds_q}&config={quote(config, safe='')}"
            f"&split={quote(split, safe='')}&offset={offset}&length={length}",
            limits.max_file_bytes)
        batch = [r["row"] for r in page.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < length:
            break
    if not rows:
        raise IngestError(f"no rows returned for {dataset}:{config}:{split}")

    column = _pick_text_column(rows, text_column)
    corpus_dir = project_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    stem_base = sanitize_filename(f"{dataset.replace('/', '-')}.txt")[:-4]
    written = 0
    for i, row in enumerate(rows):
        val = row.get(column)
        text = (val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)).strip()
        if not text:
            continue
        if _corpus_bytes(corpus_dir) + len(text) > limits.max_project_bytes:
            break
        (corpus_dir / f"{stem_base}-{i:05d}.txt").write_text(text, encoding="utf-8")
        written += 1
    return {"source": f"hf://{dataset}:{config}:{split}", "origin": "huggingface",
            "text_column": column, "rows_fetched": len(rows), "docs_written": written}
