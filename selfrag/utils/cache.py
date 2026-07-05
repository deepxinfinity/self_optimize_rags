"""Hash-keyed on-disk cache for model calls.

Every API/judge call (and, cheaply, local ollama calls) is cached to disk keyed by a hash of its
full input. This protects the $10-20 API budget across re-runs — a repeated call costs nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Optional


class DiskCache:
    def __init__(self, cache_dir: str | Path, namespace: str = "") -> None:
        self.root = Path(cache_dir) / namespace if namespace else Path(cache_dir)

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        # shard by first 2 hex chars to avoid huge flat dirs
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        p = self._path(key)
        if p.exists():
            return json.loads(p.read_text())["value"]
        return None

    def set(self, key: str, value: Any, meta: Optional[dict[str, Any]] = None) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"value": value, "meta": meta or {}}, ensure_ascii=False))

    def get_or_compute(self, payload: dict[str, Any], compute: Callable[[], Any]) -> Any:
        k = self.key(payload)
        hit = self.get(k)
        if hit is not None:
            return hit
        value = compute()
        self.set(k, value, meta={"payload_keys": sorted(payload.keys())})
        return value
