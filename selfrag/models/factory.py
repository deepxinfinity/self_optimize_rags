"""Model factory: turn a config ModelSpec into the right adapter, chosen by ``provider``.

Arms and the judge are just ModelSpecs, so they route through the same builders.
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from .base import Embedder, LLM, Reranker
from .ollama_adapter import OllamaLLM
from .st_adapter import STEmbedder, STReranker


def _spec_dict(spec: Any) -> dict[str, Any]:
    """Accept a pydantic ModelSpec/EvalArm or a plain dict; return a kwargs dict."""
    if hasattr(spec, "model_dump"):
        return spec.model_dump()
    return dict(spec)


def build_llm(spec: Any, cache_dir: str = "./.cache/api") -> LLM:
    d = _spec_dict(spec)
    provider = d.get("provider")
    if provider == "ollama":
        return OllamaLLM(**{k: v for k, v in d.items() if k != "provider"})
    if provider == "api":
        from .api_adapter import APILLM
        d.setdefault("cache_dir", cache_dir)
        return APILLM(**{k: v for k, v in d.items() if k != "provider"})
    raise ValueError(f"unknown LLM provider: {provider!r}")


def build_embedder(spec: Any) -> Embedder:
    d = _spec_dict(spec)
    provider = d.get("provider")
    if provider in ("sentence-transformers", "st"):
        return STEmbedder(**{k: v for k, v in d.items() if k != "provider"})
    raise ValueError(f"unknown embedding provider: {provider!r}")


def build_reranker(spec: Any) -> Reranker:
    d = _spec_dict(spec)
    provider = d.get("provider")
    if provider in ("sentence-transformers", "st"):
        return STReranker(**{k: v for k, v in d.items() if k != "provider"})
    raise ValueError(f"unknown reranker provider: {provider!r}")


def build_judge(spec: Any, cache_dir: str = "./.cache/api"):
    """Judge is implemented in Increment 3; the interface/factory hook exists now."""
    from .judge import build_judge_from_spec  # noqa: local import to keep Inc1 import-light
    return build_judge_from_spec(spec, cache_dir)


def build_from_config(cfg: Config) -> dict[str, Any]:
    """Convenience: build the standard set of models referenced by a config."""
    cache_dir = cfg.runtime.cache_dir
    return {
        "generator": build_llm(cfg.models.generator, cache_dir),
        "embedder": build_embedder(cfg.models.embedding),
        "reranker": build_reranker(cfg.models.reranker),  # lazy-loaded; not built until used
    }
