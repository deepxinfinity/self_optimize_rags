"""FAISS dense vector store.

Embeddings are L2-normalized upstream, so inner product (IndexFlatIP) == cosine similarity.
Exact (flat) index — the corpora here are small and correctness beats speed at this scale.
"""

from __future__ import annotations

import numpy as np

import faiss


class VectorStore:
    def __init__(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)

    def query(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return [(chunk_index, cosine_score)] for the top-k nearest chunks."""
        q = np.ascontiguousarray(query_vec, dtype=np.float32).reshape(1, -1)
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]
