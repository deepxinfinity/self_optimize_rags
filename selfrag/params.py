from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from .config import Config


@dataclass
class RetrievalParams:
    chunk_size: int
    chunk_overlap: int
    top_k: int
    hybrid_alpha: float
    use_reranker: bool
    rerank_candidates: int
    query_rewrite_enabled: bool = False


def describe(p: RetrievalParams) -> str:
    rerank = f"on ({p.rerank_candidates} candidates)" if p.use_reranker else "off"
    return (f"chunks {p.chunk_size}/{p.chunk_overlap} k={p.top_k} alpha={p.hybrid_alpha:.2f} "
            f"rerank={rerank}")


def params_from_dict(d: dict[str, Any]) -> RetrievalParams:
    # runs written before September used rerank_top_k
    candidates = d.get("rerank_candidates", d.get("rerank_top_k", d["top_k"]))
    return RetrievalParams(
        chunk_size=int(d["chunk_size"]),
        chunk_overlap=int(d["chunk_overlap"]),
        top_k=int(d["top_k"]),
        hybrid_alpha=float(d["hybrid_alpha"]),
        use_reranker=bool(d["use_reranker"]),
        rerank_candidates=int(candidates),
        query_rewrite_enabled=bool(d.get("query_rewrite_enabled", False)),
    )


def baseline_retrieval_params(cfg: Config) -> RetrievalParams:
    """The untuned reference config: `baseline:` from the config, else derived from the ranges."""
    ss = cfg.search_space
    derived = {
        "chunk_size": ss["chunk_size"]["values"][0],
        "chunk_overlap": ss["chunk_overlap"]["values"][0],
        "top_k": int(ss["top_k"]["low"]),
        "hybrid_alpha": round((ss["hybrid_alpha"]["low"] + ss["hybrid_alpha"]["high"]) / 2, 3),
        "use_reranker": False,
        "rerank_candidates": ss["rerank_candidates"]["values"][0],
        "query_rewrite_enabled": False,
    }
    known = {f.name for f in fields(RetrievalParams)}
    unknown = set(cfg.baseline) - known
    if unknown:
        raise ValueError(f"unknown keys in baseline: {sorted(unknown)}")
    return params_from_dict({**derived, **cfg.baseline})
