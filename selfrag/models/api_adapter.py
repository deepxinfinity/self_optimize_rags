"""HTTP API LLM adapter (Anthropic / OpenAI / Google) with an on-disk response cache.

Not exercised in the current local-only dev setup (no keys). Structure + caching are in place so
that swapping the judge / frontier arm to an API is a config change, not a code change. Every call
is cached by hash of its full input, so re-runs across the $10-20 budget cost nothing.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ..utils.cache import DiskCache


class APILLM:
    def __init__(self, name: str, api: str = "anthropic", cache_dir: str = "./.cache/api",
                 temperature: float = 0.0, max_tokens: Optional[int] = 1024, **_: Any) -> None:
        self.name = name
        self.api = api
        self.temperature = temperature
        self.max_tokens = max_tokens or 1024
        self.cache = DiskCache(cache_dir, namespace=f"api/{api}")

    def generate(self, prompt: str, system: Optional[str] = None, **_: Any) -> str:
        payload = {
            "api": self.api,
            "model": self.name,
            "system": system or "",
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return self.cache.get_or_compute(payload, lambda: self._call(prompt, system))

    def _call(self, prompt: str, system: Optional[str]) -> str:
        if self.api == "anthropic":
            return self._anthropic(prompt, system)
        raise NotImplementedError(
            f"API provider {self.api!r} not wired yet; only 'anthropic' is implemented. "
            "Add its branch in APILLM._call."
        )

    def _anthropic(self, prompt: str, system: Optional[str]) -> str:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set — required for the api provider.")
        try:
            import anthropic  # lazy: only needed when an API model is actually used
        except ImportError as e:
            raise RuntimeError("`anthropic` package not installed; add it to requirements.") from e
        client = anthropic.Anthropic()
        kwargs: dict[str, Any] = {
            "model": self.name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")
