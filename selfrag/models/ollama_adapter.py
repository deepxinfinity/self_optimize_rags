from __future__ import annotations

from typing import Any, Iterable, Optional

import httpx
import ollama

# Long enough for a cold model load plus a full answer; without a limit a stopped server
# (Ctrl+Z on `ollama serve` keeps the port open) makes a run wait forever.
DEFAULT_TIMEOUT = 600.0

_NOT_ANSWERING = ("ollama is not answering. Is `ollama serve` running in another terminal, and "
                  "not suspended (Ctrl+Z stops it; type `fg` in that terminal to resume)?")


class OllamaUnavailable(RuntimeError):
    pass


class ModelOutputError(RuntimeError):
    """ollama cut the reply off, e.g. the model got stuck repeating a token."""


class OllamaLLM:
    def __init__(self, name: str, temperature: float = 0.0, max_tokens: Optional[int] = None,
                 num_ctx: Optional[int] = None, timeout: float = DEFAULT_TIMEOUT,
                 **_: Any) -> None:
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        # ollama silently drops the start of prompts longer than its default window
        self.num_ctx = num_ctx
        self.timeout = timeout
        self._client = ollama.Client(timeout=timeout)

    def _options(self, **overrides: Any) -> dict[str, Any]:
        opts: dict[str, Any] = {"temperature": self.temperature}
        if self.max_tokens is not None:
            opts["num_predict"] = self.max_tokens
        if self.num_ctx is not None:
            opts["num_ctx"] = self.num_ctx
        opts.update({k: v for k, v in overrides.items() if v is not None})
        return opts

    def generate(self, prompt: str, system: Optional[str] = None,
                 format: Optional[str] = None, **options: Any) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self._client.chat(
                model=self.name,
                messages=messages,
                options=self._options(**options),
                format=format or "",
            )
        except httpx.TimeoutException as e:
            if _server_answers():
                raise OllamaUnavailable(
                    f"{self.name}: still generating after {self.timeout:.0f}s while the server "
                    "is up, so the model is probably stuck in a loop. Set max_tokens for this "
                    "model in the config.") from e
            raise OllamaUnavailable(f"{self.name}: no answer within {self.timeout:.0f}s. "
                                    f"{_NOT_ANSWERING}") from e
        except ConnectionError as e:
            raise OllamaUnavailable(f"{self.name}: {_NOT_ANSWERING}") from e
        except ollama.ResponseError as e:
            if "prediction aborted" in e.error or "repeat limit" in e.error:
                raise ModelOutputError(f"{self.name}: {e.error}") from e
            raise
        return resp["message"]["content"]


def _server_answers(timeout: float = 5.0) -> bool:
    try:
        ollama.Client(timeout=timeout).list()
    except (httpx.HTTPError, ConnectionError):
        return False
    return True


def check_ollama(models: Iterable[str], timeout: float = 10.0) -> None:
    """Fail fast if the server is down or stopped, or a model hasn't been pulled."""
    try:
        listed = ollama.Client(timeout=timeout).list()
    except (httpx.TimeoutException, ConnectionError) as e:
        raise OllamaUnavailable(_NOT_ANSWERING) from e
    have = {m["model"] for m in listed["models"]}
    missing = sorted({m for m in models if m not in have and f"{m}:latest" not in have})
    if missing:
        raise OllamaUnavailable(f"models not pulled: {', '.join(missing)} "
                                f"(run `ollama pull <name>`)")
