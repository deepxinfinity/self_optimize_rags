from __future__ import annotations

from typing import Any

from .base import Embedder, LLM, Reranker
from .ollama_adapter import OllamaLLM
from .st_adapter import STEmbedder, STReranker


def _spec_dict(spec: Any) -> dict[str, Any]:
    if hasattr(spec, "model_dump"):
        return spec.model_dump()
    return dict(spec)


def build_llm(spec: Any, cache_dir: str = "./.cache/api") -> LLM:
    d = _spec_dict(spec)
    provider = d.pop("provider", None)
    if provider == "ollama":
        return OllamaLLM(**d)
    if provider == "api":
        from .api_adapter import APILLM
        d.setdefault("cache_dir", cache_dir)
        return APILLM(**d)
    raise ValueError(f"unknown LLM provider: {provider!r}")


def build_embedder(spec: Any) -> Embedder:
    d = _spec_dict(spec)
    provider = d.pop("provider", None)
    if provider in ("sentence-transformers", "st"):
        return STEmbedder(**d)
    raise ValueError(f"unknown embedding provider: {provider!r}")


def build_reranker(spec: Any) -> Reranker:
    d = _spec_dict(spec)
    provider = d.pop("provider", None)
    if provider in ("sentence-transformers", "st"):
        return STReranker(**d)
    raise ValueError(f"unknown reranker provider: {provider!r}")


def build_judge(spec: Any, cache_dir: str = "./.cache/api"):
    from .judge import build_judge_from_spec
    return build_judge_from_spec(spec, cache_dir)
