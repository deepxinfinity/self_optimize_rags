"""Dense + BM25 retrieval blended with a tunable alpha.

Both score vectors are min-max normalised before mixing:
    score = alpha * dense + (1 - alpha) * bm25
so alpha=1 is dense only and alpha=0 is BM25 only.
"""

from __future__ import annotations

import numpy as np

from .bm25 import BM25Index
from .vector_store import VectorStore


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class HybridRetriever:
    """Both indices must be built over the same chunk list, in the same order."""

    def __init__(self, vector_store: VectorStore, bm25: BM25Index) -> None:
        self.vs = vector_store
        self.bm25 = bm25
        self.n = vector_store.n

    def retrieve(self, query_vec: np.ndarray, query_text: str, top_k: int,
                 alpha: float) -> list[tuple[int, float]]:
        if alpha >= 1.0:
            combined = _minmax(self.vs.scores(query_vec))
        elif alpha <= 0.0:
            combined = _minmax(self.bm25.scores(query_text))
        else:
            combined = (alpha * _minmax(self.vs.scores(query_vec))
                        + (1.0 - alpha) * _minmax(self.bm25.scores(query_text)))

        k = min(top_k, self.n)
        top = np.argsort(-combined, kind="stable")[:k]
        return [(int(i), float(combined[i])) for i in top]
