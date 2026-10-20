"""On-disk cache for model calls, keyed by a hash of the full input."""

from __future__ import annotations

import hashlib
import json
import os
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
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        p = self._path(key)
        try:
            return json.loads(p.read_text())["value"]
        except (OSError, json.JSONDecodeError, KeyError):
            return None

    def set(self, key: str, value: Any, meta: Optional[dict[str, Any]] = None) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # write-then-rename so a run killed mid-write can't leave a truncated entry behind
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"value": value, "meta": meta or {}}, ensure_ascii=False))
        os.replace(tmp, p)

    def get_or_compute(self, payload: dict[str, Any], compute: Callable[[], Any]) -> Any:
        k = self.key(payload)
        hit = self.get(k)
        if hit is not None:
            return hit
        value = compute()
        self.set(k, value, meta={"payload_keys": sorted(payload.keys())})
        return value


class CachedLLM:
    """Caches a deterministic (temperature 0) model, so a restarted run doesn't regenerate
    every answer it already has."""

    def __init__(self, llm: Any, cache: DiskCache) -> None:
        self.llm = llm
        self.cache = cache
        self.name = getattr(llm, "name", "")

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> str:
        payload = {
            "model": self.name,
            "system": system or "",
            "prompt": prompt,
            "options": {k: getattr(self.llm, k, None)
                        for k in ("temperature", "max_tokens", "num_ctx")},
            "kwargs": kwargs,
        }
        return self.cache.get_or_compute(
            payload, lambda: self.llm.generate(prompt, system=system, **kwargs))
