"""Timing instrumentation — a first-class deliverable, not debug-only.

Two granularities (see notes/decisions-log.md):
  1. Per config/trial: the optimizer records a ``time_sec`` next to each trial's scores.
  2. Per pipeline step: chunking, index build, embedding, retrieve, rerank, generation, ...
     collected here and dumped to ``runs/<timestamp>/timings.json``.

Kept deliberately cheap: a monotonic clock and a list of records, negligible overhead.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class TimingLog:
    """Collects labelled step durations. Nestable via repeated ``timed`` calls."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    @contextmanager
    def timed(self, label: str, **meta: Any) -> Iterator[dict[str, Any]]:
        """Time a block. ``meta`` (e.g. n_chunks=...) is stored alongside the duration."""
        rec: dict[str, Any] = {"label": label, **meta}
        t0 = time.perf_counter()
        try:
            yield rec
        finally:
            rec["seconds"] = round(time.perf_counter() - t0, 4)
            self.records.append(rec)

    def total(self) -> float:
        return round(sum(r.get("seconds", 0.0) for r in self.records), 4)

    def as_dict(self) -> dict[str, Any]:
        return {"steps": self.records, "total_seconds": self.total()}

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2))

    def summary(self) -> str:
        lines = [f"  {r['seconds']:>8.3f}s  {r['label']}" for r in self.records]
        lines.append(f"  {'-' * 8}")
        lines.append(f"  {self.total():>8.3f}s  total")
        return "\n".join(lines)


@contextmanager
def timed(label: str = "") -> Iterator[dict[str, float]]:
    """Standalone one-off timer: ``with timed() as t: ...`` then read ``t['seconds']``."""
    rec: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        yield rec
    finally:
        rec["seconds"] = round(time.perf_counter() - t0, 4)
