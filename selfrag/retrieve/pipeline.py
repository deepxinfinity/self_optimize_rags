from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..corpus.chunker import Chunk
from ..index.hybrid import HybridRetriever
from ..models.base import Embedder, Reranker
from ..params import RetrievalParams
from ..utils.timing import TimingLog, maybe_timed


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


class RetrievePipeline:
    """query -> (rewrite) -> hybrid search -> (rerank) -> top_k -> trim to the context budget."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder, hybrid: HybridRetriever,
                 reranker: Optional[Reranker] = None, rewriter=None,
                 context_budget: Optional[int] = None) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.hybrid = hybrid
        self.reranker = reranker
        self.rewriter = rewriter
        self.context_budget = context_budget

    def retrieve(self, query: str, params: RetrievalParams,
                 timing: Optional[TimingLog] = None) -> list[RetrievalResult]:
        if params.query_rewrite_enabled and self.rewriter is not None:
            query = self.rewriter.rewrite(query, timing)

        rerank = params.use_reranker and self.reranker is not None
        first_k = max(params.top_k, params.rerank_candidates) if rerank else params.top_k
        with maybe_timed(timing, "retrieve", top_k=first_k, alpha=params.hybrid_alpha):
            qvec = self.embedder.embed([query])[0]
            hits = self.hybrid.retrieve(qvec, query, top_k=first_k, alpha=params.hybrid_alpha)
        results = [RetrievalResult(self.chunks[i], s) for i, s in hits]

        if rerank and results:
            with maybe_timed(timing, "rerank", n_candidates=len(results)):
                scores = self.reranker.rerank(query, [r.chunk.text for r in results])
            results = sorted((RetrievalResult(r.chunk, s) for r, s in zip(results, scores)),
                             key=lambda r: r.score, reverse=True)
        return self._fit_budget(results[:params.top_k])

    def _fit_budget(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        # A fixed word budget keeps configs comparable: without it, bigger chunks and larger
        # top_k always win on recall simply by showing the answerer more of the document.
        if not self.context_budget:
            return results
        kept: list[RetrievalResult] = []
        used = 0
        for r in results:
            n = len(r.chunk.text.split())
            if kept and used + n > self.context_budget:
                break
            kept.append(r)
            used += n
        return kept
