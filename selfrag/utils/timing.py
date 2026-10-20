from __future__ import annotations

import json
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, ContextManager, Iterator, Optional


class TimingLog:
    """Labelled step durations, dumped to timings.json at the end of a run."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._depth = 0

    def _new(self, label: str, meta: dict[str, Any]) -> dict[str, Any]:
        rec: dict[str, Any] = {"label": label, **meta}
        if self._depth:
            rec["depth"] = self._depth
        return rec

    @contextmanager
    def timed(self, label: str, **meta: Any) -> Iterator[dict[str, Any]]:
        rec = self._new(label, meta)
        self._depth += 1
        t0 = time.perf_counter()
        try:
            yield rec
        finally:
            self._depth -= 1
            rec["seconds"] = round(time.perf_counter() - t0, 4)
            self.records.append(rec)

    def record(self, label: str, seconds: float = 0.0, **meta: Any) -> None:
        rec = self._new(label, meta)
        rec["seconds"] = round(seconds, 4)
        self.records.append(rec)

    def count(self, label: str) -> int:
        return sum(1 for r in self.records if r.get("label") == label)

    def total(self) -> float:
        # nested steps are already inside their parent's time
        return round(sum(r["seconds"] for r in self.records if not r.get("depth")), 4)

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


def maybe_timed(timing: Optional[TimingLog], label: str, **meta: Any) -> ContextManager:
    if timing is None:
        return nullcontext({})
    return timing.timed(label, **meta)


@contextmanager
def timed(label: str = "") -> Iterator[dict[str, float]]:
    rec: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        yield rec
    finally:
        rec["seconds"] = round(time.perf_counter() - t0, 4)
