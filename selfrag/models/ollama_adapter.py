"""Ollama-backed LLM adapter (local generation)."""

from __future__ import annotations

from typing import Any, Optional

import ollama


class OllamaLLM:
    def __init__(self, name: str, temperature: float = 0.0, max_tokens: Optional[int] = None,
                 **_: Any) -> None:
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _options(self, **overrides: Any) -> dict[str, Any]:
        opts: dict[str, Any] = {"temperature": self.temperature}
        if self.max_tokens is not None:
            opts["num_predict"] = self.max_tokens
        opts.update({k: v for k, v in overrides.items() if v is not None})
        return opts

    def generate(self, prompt: str, system: Optional[str] = None,
                 format: Optional[str] = None, **options: Any) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = ollama.chat(
            model=self.name,
            messages=messages,
            options=self._options(**options),
            format=format or "",
        )
        return resp["message"]["content"]
