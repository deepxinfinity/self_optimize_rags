from __future__ import annotations

import numpy as np

from conftest import CountingEmbedder, params

from selfrag.corpus.chunker import Chunk
from selfrag.index.bm25 import BM25Index
from selfrag.index.hybrid import HybridRetriever
from selfrag.index.vector_store import VectorStore
from selfrag.retrieve.pipeline import RetrievePipeline


def _pipeline(n_chunks: int = 30, words: int = 100, reranker=None, budget=None):
    chunks = [Chunk(f"d.txt::{i}", "d.txt", " ".join([f"topic{i}"] * words), i)
              for i in range(n_chunks)]
    emb = CountingEmbedder()
    texts = [c.text for c in chunks]
    hybrid = HybridRetriever(VectorStore(emb.embed(texts)), BM25Index(texts))
    return RetrievePipeline(chunks, emb, hybrid, reranker=reranker, context_budget=budget)


class _PreferHighOrdinal:
    def __init__(self) -> None:
        self.seen = 0

    def rerank(self, query, passages):
        self.seen = len(passages)
        return [float(int(p.split()[0].removeprefix("topic"))) for p in passages]


def test_top_k_without_reranker():
    out = _pipeline().retrieve("topic3", params(top_k=4))
    assert len(out) == 4
    assert out[0].chunk.ordinal == 3


def test_reranker_scores_the_whole_candidate_pool_then_keeps_top_k():
    rr = _PreferHighOrdinal()
    out = _pipeline(reranker=rr).retrieve("topic3", params(top_k=3, use_reranker=True,
                                                            rerank_candidates=20))
    assert rr.seen == 20
    assert len(out) == 3
    assert [r.chunk.ordinal for r in out] == sorted((r.chunk.ordinal for r in out), reverse=True)


def test_reranker_ignored_when_disabled_in_params():
    rr = _PreferHighOrdinal()
    _pipeline(reranker=rr).retrieve("topic3", params(top_k=3, use_reranker=False))
    assert rr.seen == 0


def test_context_budget_trims_by_words():
    out = _pipeline(words=100, budget=250).retrieve("topic3", params(top_k=10))
    assert len(out) == 2


def test_context_budget_always_keeps_the_first_chunk():
    out = _pipeline(words=100, budget=50).retrieve("topic3", params(top_k=10))
    assert len(out) == 1


def test_no_budget_means_top_k():
    out = _pipeline(words=100, budget=None).retrieve("topic3", params(top_k=10))
    assert len(out) == 10


def test_hybrid_alpha_extremes_are_pure_signals():
    p = _pipeline()
    qvec = p.embedder.embed(["topic5"])[0]
    dense_only = p.hybrid.retrieve(qvec, "topic5", top_k=1, alpha=1.0)
    bm25_only = p.hybrid.retrieve(qvec, "topic5", top_k=1, alpha=0.0)
    assert dense_only[0][0] == 5 and bm25_only[0][0] == 5
    assert np.isclose(dense_only[0][1], 1.0)
