from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import Config
from .corpus.chunker import Chunk, chunk_documents
from .corpus.loader import Document, load_corpus
from .generate.answerer import Answerer
from .index.bm25 import BM25Index
from .index.hybrid import HybridRetriever
from .index.vector_store import VectorStore
from .models.factory import build_embedder, build_llm, build_reranker
from .params import RetrievalParams, baseline_retrieval_params
from .retrieve.pipeline import RetrievalResult, RetrievePipeline
from .utils.cache import CachedLLM, DiskCache
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


class RagFactory:
    """Builds RAGs over one corpus.

    Indices depend only on (chunk_size, chunk_overlap), so each chunking is embedded once per
    factory and, with runtime.embedding_cache_dir set, once ever. Models load lazily, so a run
    that never reranks never loads the cross-encoder. Not thread-safe.
    """

    def __init__(self, cfg: Config, timing: Optional[TimingLog] = None) -> None:
        self.cfg = cfg
        self.timing = timing or TimingLog()
        self.context_budget = cfg.retrieval.get("context_budget_words")
        cache_dir = cfg.runtime.embedding_cache_dir
        self.embedding_cache = Path(cache_dir) if cache_dir else None
        self._docs: Optional[list[Document]] = None
        self._embedder = None
        self._reranker = None
        self._generator = None
        self._indices: dict[tuple[int, int], tuple[list[Chunk], HybridRetriever]] = {}

    @property
    def docs(self) -> list[Document]:
        if self._docs is None:
            with self.timing.timed("load_corpus") as rec:
                self._docs = load_corpus(self.cfg.dataset.corpus_path, self.cfg.dataset.corpus_glob)
                rec["n_docs"] = len(self._docs)
        return self._docs

    @property
    def embedder(self):
        if self._embedder is None:
            with self.timing.timed("load_embedder"):
                self._embedder = build_embedder(self.cfg.models.embedding)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            with self.timing.timed("load_reranker"):
                self._reranker = build_reranker(self.cfg.models.reranker)
        return self._reranker

    @property
    def generator(self):
        if self._generator is None:
            spec = self.cfg.models.generator
            llm = build_llm(spec, self.cfg.runtime.cache_dir)
            # api models cache themselves; sampled outputs shouldn't be cached at all
            if spec.provider != "api" and getattr(llm, "temperature", None) == 0:
                llm = CachedLLM(llm, DiskCache(self.cfg.runtime.cache_dir, "generator"))
            self._generator = llm
        return self._generator

    @property
    def n_indices_built(self) -> int:
        return len(self._indices)

    def _embed(self, texts: list[str]) -> np.ndarray:
        path = None
        if self.embedding_cache is not None:
            digest = hashlib.sha256(self.cfg.models.embedding.name.encode())
            for t in texts:
                digest.update(b"\0" + t.encode())
            path = self.embedding_cache / f"{digest.hexdigest()[:32]}.npy"
            if path.exists():
                self.timing.record("embed_cache_hit", n_chunks=len(texts))
                return np.load(path)

        with self.timing.timed("embed", n_chunks=len(texts)):
            vectors = self.embedder.embed(texts)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, vectors)
            os.replace(tmp, path)
        return vectors

    def index_for(self, chunk_size: int, chunk_overlap: int):
        key = (chunk_size, chunk_overlap)
        if key in self._indices:
            self.timing.record("index_cache_hit", chunk_size=chunk_size, overlap=chunk_overlap)
            return self._indices[key]
        with self.timing.timed("chunk", chunk_size=chunk_size, overlap=chunk_overlap) as rec:
            chunks = chunk_documents(self.docs, chunk_size, chunk_overlap)
            rec["n_chunks"] = len(chunks)
        texts = [c.text for c in chunks]
        embeddings = self._embed(texts)
        with self.timing.timed("build_index", n_chunks=len(chunks)):
            hybrid = HybridRetriever(VectorStore(embeddings), BM25Index(texts))
        self._indices[key] = (chunks, hybrid)
        return self._indices[key]

    def build(self, params: Optional[RetrievalParams] = None,
              answer_prompt: Optional[str] = None) -> RAG:
        params = params or baseline_retrieval_params(self.cfg)
        chunks, hybrid = self.index_for(params.chunk_size, params.chunk_overlap)

        prompts = self.cfg.search_space["prompts"]
        rewriter = None
        if params.query_rewrite_enabled:
            from .generate.rewriter import Rewriter
            rewriter = Rewriter(self.generator, prompts["rewrite_prompt_seed"])

        pipeline = RetrievePipeline(chunks, self.embedder, hybrid,
                                    self.reranker if params.use_reranker else None, rewriter,
                                    context_budget=self.context_budget)
        answerer = Answerer(self.generator, answer_prompt or prompts["answer_prompt_seed"])
        return RAG(self.cfg, params, pipeline, answerer, chunks, self.timing)


def build_rag(cfg: Config, params: Optional[RetrievalParams] = None,
              timing: Optional[TimingLog] = None, answer_prompt: Optional[str] = None) -> RAG:
    return RagFactory(cfg, timing).build(params, answer_prompt)
