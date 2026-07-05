"""Retrieval pipeline: query -> (optional rewrite) -> hybrid retrieve -> (optional rerank) -> chunks.

The pipeline holds the built indices and the chunk list; a single ``retrieve`` call takes the
per-query knobs from a RetrievalParams. Query rewriting is a no-op hook here (wired in Increment 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..corpus.chunker import Chunk
from ..index.hybrid import HybridRetriever
from ..models.base import Embedder, Reranker
from ..params import RetrievalParams


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


class RetrievePipeline:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, hybrid: HybridRetriever,
                 reranker: Optional[Reranker] = None, rewriter=None) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.hybrid = hybrid
        self.reranker = reranker
        self.rewriter = rewriter

    def retrieve(self, query: str, params: RetrievalParams,
                 timing=None) -> list[RetrievalResult]:
        # Optional query rewrite (Stage-2 knob); dormant unless enabled and a rewriter is present.
        if params.query_rewrite_enabled and self.rewriter is not None:
            query = self.rewriter.rewrite(query, timing)

        # First-stage retrieval favors recall; if reranking, pull a wider candidate set.
        first_k = max(params.top_k, params.rerank_top_k) if params.use_reranker else params.top_k

        def _dense_then_hybrid():
            qvec = self.embedder.embed([query])[0]
            return self.hybrid.retrieve(qvec, query, top_k=first_k, alpha=params.hybrid_alpha)

        if timing is not None:
            with timing.timed("retrieve", top_k=first_k, alpha=params.hybrid_alpha):
                hits = _dense_then_hybrid()
        else:
            hits = _dense_then_hybrid()

        results = [RetrievalResult(self.chunks[i], s) for i, s in hits]

        if params.use_reranker and self.reranker is not None and results:
            results = self._rerank(query, results, params.rerank_top_k, timing)
        return results[:params.top_k]

    def _rerank(self, query: str, results: list[RetrievalResult], rerank_top_k: int,
                timing) -> list[RetrievalResult]:
        passages = [r.chunk.text for r in results]

        def _do():
            scores = self.reranker.rerank(query, passages)
            reranked = [RetrievalResult(r.chunk, s) for r, s in zip(results, scores)]
            reranked.sort(key=lambda r: r.score, reverse=True)
            return reranked[:rerank_top_k]

        if timing is not None:
            with timing.timed("rerank", n_candidates=len(passages)):
                return _do()
        return _do()
