"""Hybrid dense + BM25 retrieval with a tunable alpha.

Dense (cosine) and BM25 scores live on different scales, so each is min-max normalized across the
candidate set before combining:

    score = alpha * dense_norm + (1 - alpha) * bm25_norm

alpha=1.0 -> dense only, alpha=0.0 -> BM25 only (matches configs/default.yaml's hybrid_alpha).
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
    """Owns both indices over the *same* ordered chunk list and blends their scores."""

    def __init__(self, vector_store: VectorStore, bm25: BM25Index) -> None:
        self.vs = vector_store
        self.bm25 = bm25
        self.n = vector_store.index.ntotal

    def retrieve(self, query_vec: np.ndarray, query_text: str, top_k: int,
                 alpha: float) -> list[tuple[int, float]]:
        # Dense scores over all chunks (small corpora -> cheap to score everything).
        dense = np.zeros(self.n, dtype=np.float32)
        for idx, score in self.vs.query(query_vec, self.n):
            dense[idx] = score

        if alpha >= 1.0:
            combined = _minmax(dense)
        elif alpha <= 0.0:
            combined = _minmax(self.bm25.scores(query_text))
        else:
            combined = alpha * _minmax(dense) + (1.0 - alpha) * _minmax(self.bm25.scores(query_text))

        k = min(top_k, self.n)
        top = np.argsort(-combined)[:k]
        return [(int(i), float(combined[i])) for i in top]
