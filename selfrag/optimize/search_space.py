"""Turn the config's search_space ranges into Optuna suggestions.

Ranges live in the config (nothing hardcoded here). Stage 1 tunes the retrieval knobs only;
``query_rewrite_enabled`` and the prompt seeds are Stage-2 concerns and are left at their defaults.
"""

from __future__ import annotations

from typing import Any

from ..params import RetrievalParams

# The retrieval knobs Stage 1 searches over, in config-key form.
STAGE1_KNOBS = ("chunk_size", "chunk_overlap", "top_k", "hybrid_alpha",
                "use_reranker", "rerank_top_k")


def _suggest(trial: Any, name: str, spec: dict[str, Any]) -> Any:
    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["values"])
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    if kind == "float":
        return trial.suggest_float(name, spec["low"], spec["high"])
    raise ValueError(f"unknown search-space type for {name!r}: {kind!r}")


def suggest_retrieval_params(trial: Any, search_space: dict[str, Any]) -> RetrievalParams:
    vals = {k: _suggest(trial, k, search_space[k]) for k in STAGE1_KNOBS}
    return params_from_dict(vals)


def params_from_dict(d: dict[str, Any]) -> RetrievalParams:
    """Build RetrievalParams from a knob dict (e.g. Optuna's ``study.best_params``)."""
    return RetrievalParams(
        chunk_size=int(d["chunk_size"]),
        chunk_overlap=int(d["chunk_overlap"]),
        top_k=int(d["top_k"]),
        hybrid_alpha=float(d["hybrid_alpha"]),
        use_reranker=bool(d["use_reranker"]),
        rerank_top_k=int(d["rerank_top_k"]),
        query_rewrite_enabled=bool(d.get("query_rewrite_enabled", False)),
    )
