from __future__ import annotations

from typing import Any

from ..params import RetrievalParams, params_from_dict

STAGE1_KNOBS = ("chunk_size", "chunk_overlap", "top_k", "hybrid_alpha",
                "use_reranker", "rerank_candidates")


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
    return params_from_dict({k: _suggest(trial, k, search_space[k]) for k in STAGE1_KNOBS})


__all__ = ["STAGE1_KNOBS", "params_from_dict", "suggest_retrieval_params"]
