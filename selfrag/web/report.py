"""Corpus report shown after ingestion: sizes, duplicates, warnings and rough cost estimates."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

# bge-base on an M-series Mac; replaced by the project's own timings once a run exists
_DEFAULT_EMBED_SEC_PER_CHUNK = 0.04
_EMBED_DIM = 768
_TINY_CHARS = 200
_HUGE_CHARS = 1_000_000

_WS = re.compile(r"\s+")

_HIST_BUCKETS = [(0, 500), (500, 2_000), (2_000, 10_000), (10_000, 50_000), (50_000, None)]


def _bucket_label(lo: int, hi: Optional[int]) -> str:
    return f"{lo:,}–{hi:,}" if hi else f">{lo:,}"


def _embed_rate_from_runs(runs_dir: Path) -> Optional[float]:
    if not runs_dir.exists():
        return None
    for run in sorted(runs_dir.iterdir(), reverse=True):
        t = run / "timings.json"
        if not t.exists():
            continue
        try:
            steps = json.loads(t.read_text()).get("steps", [])
        except (OSError, json.JSONDecodeError):
            continue
        for step in steps:
            if step.get("label") == "embed" and step.get("n_chunks") and step.get("seconds"):
                return step["seconds"] / step["n_chunks"]
    return None


def corpus_report(project_dir: Path, chunk_size: int = 256, chunk_overlap: int = 50) -> dict[str, Any]:
    corpus_dir = project_dir / "corpus"
    files = sorted(p for p in corpus_dir.rglob("*") if p.is_file()
                   and p.suffix.lower() in (".txt", ".md")) if corpus_dir.exists() else []

    lengths: list[int] = []
    words_total = 0
    hashes: dict[str, list[str]] = {}
    warnings: list[str] = []

    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        n = len(text)
        lengths.append(n)
        words_total += len(text.split())
        if n == 0:
            warnings.append(f"{p.name}: empty after normalization")
        elif n < _TINY_CHARS:
            warnings.append(f"{p.name}: very short ({n} chars), may add noise")
        elif n > _HUGE_CHARS:
            warnings.append(f"{p.name}: very large ({n:,} chars), consider splitting")
        digest = hashlib.md5(_WS.sub(" ", text.lower()).encode()).hexdigest()
        hashes.setdefault(digest, []).append(p.name)

    duplicates = [names for names in hashes.values() if len(names) > 1]
    histogram = [
        {"bucket": _bucket_label(lo, hi),
         "count": sum(1 for n in lengths if n >= lo and (hi is None or n < hi))}
        for lo, hi in _HIST_BUCKETS
    ]

    step = max(chunk_size - chunk_overlap, 1)
    est_chunks = max((words_total + step - 1) // step, len(files)) if words_total else 0
    rate = _embed_rate_from_runs(project_dir / "runs") or _DEFAULT_EMBED_SEC_PER_CHUNK

    return {
        "n_docs": len(files),
        "total_chars": sum(lengths),
        "total_words": words_total,
        "est_tokens": int(words_total * 1.3),
        "length_histogram_chars": histogram,
        "duplicate_groups": duplicates,
        "warnings": warnings,
        "estimates": {
            "note": "rough estimates; embed rate refined from this project's timing logs when available",
            "chunk_size_words": chunk_size,
            "est_chunks": est_chunks,
            "est_index_mb": round(est_chunks * _EMBED_DIM * 4 / 1e6, 1),
            "est_embed_seconds": round(est_chunks * rate),
        },
    }
