"""Retrieval/prompt parameters and a baseline derived from the config search space.

The optimizer (Increment 4+) will sample these from ``search_space``; for the naive baseline we
derive a single deterministic point. Numeric knobs come from the config's allowed ranges (nothing
hardcoded); the on/off toggles are fixed to their *naive* setting (no reranker, no query rewrite)
so the baseline is a genuine untuned reference for the optimizer to beat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Config


@dataclass
class RetrievalParams:
    chunk_size: int
    chunk_overlap: int
    top_k: int
    hybrid_alpha: float
    use_reranker: bool
    rerank_top_k: int
    query_rewrite_enabled: bool = False


def _first_categorical(spec: dict[str, Any]) -> Any:
    return spec["values"][0]


def baseline_retrieval_params(cfg: Config) -> RetrievalParams:
    ss = cfg.search_space
    return RetrievalParams(
        chunk_size=_first_categorical(ss["chunk_size"]),
        chunk_overlap=_first_categorical(ss["chunk_overlap"]),
        top_k=int(ss["top_k"]["low"]),
        # Neutral hybrid midpoint as the untuned starting point.
        hybrid_alpha=round((ss["hybrid_alpha"]["low"] + ss["hybrid_alpha"]["high"]) / 2, 3),
        use_reranker=False,                                   # naive: off
        rerank_top_k=_first_categorical(ss["rerank_top_k"]),
        query_rewrite_enabled=False,                          # naive: off
    )
