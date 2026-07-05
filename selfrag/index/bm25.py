"""BM25 sparse keyword retrieval over chunk texts (rank_bm25)."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, texts: list[str]) -> None:
        self.corpus_tokens = [_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def scores(self, query: str) -> np.ndarray:
        """Return a BM25 score per chunk, aligned with construction order."""
        return np.asarray(self.bm25.get_scores(_tokenize(query)), dtype=np.float32)

    def query(self, query: str, k: int) -> list[tuple[int, float]]:
        s = self.scores(query)
        k = min(k, len(s))
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top]
