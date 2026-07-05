"""End-to-end RAG assembly: wire corpus -> chunk -> embed -> index -> retrieve -> answer.

Everything is driven by a Config plus a RetrievalParams (the baseline, or a point the optimizer is
evaluating). Each stage is timed into a TimingLog so we can see where wall-clock time goes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .corpus.chunker import Chunk, chunk_documents
from .corpus.loader import load_corpus
from .generate.answerer import Answerer
from .index.bm25 import BM25Index
from .index.hybrid import HybridRetriever
from .index.vector_store import VectorStore
from .models.factory import build_embedder, build_llm, build_reranker
from .params import RetrievalParams, baseline_retrieval_params
from .retrieve.pipeline import RetrievalResult, RetrievePipeline
from .utils.timing import TimingLog


@dataclass
class AnswerResult:
    answer: str
    sources: list[RetrievalResult]


class RAG:
    def __init__(self, cfg: Config, params: RetrievalParams, pipeline: RetrievePipeline,
                 answerer: Answerer, chunks: list[Chunk], timing: TimingLog) -> None:
        self.cfg = cfg
        self.params = params
        self.pipeline = pipeline
        self.answerer = answerer
        self.chunks = chunks
        self.timing = timing

    def ask(self, question: str) -> AnswerResult:
        results = self.pipeline.retrieve(question, self.params, self.timing)
        answer = self.answerer.answer(question, results, self.timing)
        return AnswerResult(answer=answer, sources=results)


def build_rag(cfg: Config, params: RetrievalParams | None = None,
              timing: TimingLog | None = None) -> RAG:
    timing = timing or TimingLog()
    params = params or baseline_retrieval_params(cfg)

    with timing.timed("load_corpus") as rec:
        docs = load_corpus(cfg.dataset.corpus_path, cfg.dataset.corpus_glob)
        rec["n_docs"] = len(docs)

    with timing.timed("chunk", chunk_size=params.chunk_size, overlap=params.chunk_overlap) as rec:
        chunks = chunk_documents(docs, params.chunk_size, params.chunk_overlap)
        rec["n_chunks"] = len(chunks)

    embedder = build_embedder(cfg.models.embedding)
    with timing.timed("embed", n_chunks=len(chunks)):
        embeddings = embedder.embed([c.text for c in chunks])

    with timing.timed("build_index"):
        vector_store = VectorStore(embeddings)
        bm25 = BM25Index([c.text for c in chunks])
        hybrid = HybridRetriever(vector_store, bm25)

    reranker = build_reranker(cfg.models.reranker) if params.use_reranker else None
    generator = build_llm(cfg.models.generator, cfg.runtime.cache_dir)
    prompts = cfg.search_space["prompts"]
    answer_prompt = prompts["answer_prompt_seed"]

    rewriter = None
    if params.query_rewrite_enabled:
        from .generate.rewriter import Rewriter
        rewriter = Rewriter(generator, prompts["rewrite_prompt_seed"])

    pipeline = RetrievePipeline(chunks, embedder, hybrid, reranker, rewriter)
    answerer = Answerer(generator, answer_prompt)
    return RAG(cfg, params, pipeline, answerer, chunks, timing)
