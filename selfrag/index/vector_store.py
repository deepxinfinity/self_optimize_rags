"""Exact dense search in numpy. Embeddings are unit length, so scores are cosine similarity.

This used to wrap a flat FAISS index, but the hybrid retriever needs a score for every chunk
anyway, and at ~20k chunks that is one matrix-vector product. FAISS and torch also ship separate
OpenMP runtimes that abort the process on macOS when FAISS initialises first.
"""

from __future__ import annotations

import numpy as np


class VectorStore:
    def __init__(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.dim = self.embeddings.shape[1]

    @property
    def n(self) -> int:
        return self.embeddings.shape[0]

    def scores(self, query_vec: np.ndarray) -> np.ndarray:
        return self.embeddings @ np.asarray(query_vec, dtype=np.float32).reshape(-1)

    def query(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        s = self.scores(query_vec)
        k = min(k, s.size)
        if k <= 0:
            return []
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return [(int(i), float(s[i])) for i in top]
