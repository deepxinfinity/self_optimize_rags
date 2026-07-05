"""Stage 1: optimize retrieval knobs against retrieval metrics (cheap — no generation).

Optuna searches chunk_size / overlap / top_k / hybrid_alpha / reranker on-off / rerank_top_k,
scored by hit_rate or MRR on the *val* split. The winning config is frozen for Stage 2.

Efficiency: embedding the corpus is the slow step, so built indices are cached per
(chunk_size, chunk_overlap) — there are only a handful of chunking combos, and most trials reuse
one. Each trial records its wall-clock ``time_sec`` (the timing-analysis requirement) into a
per-trial JSONL log.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import optuna
from optuna.samplers import TPESampler

from ..config import Config
from ..corpus.chunker import Chunk, chunk_documents
from ..corpus.loader import load_corpus
from ..evalgen.generator import QAPair
from ..index.bm25 import BM25Index
from ..index.hybrid import HybridRetriever
from ..index.vector_store import VectorStore
from ..metrics.retrieval import retrieval_metrics
from ..models.factory import build_embedder, build_reranker
from ..params import RetrievalParams, baseline_retrieval_params
from ..retrieve.pipeline import RetrievePipeline
from ..utils.runs import append_jsonl
from .search_space import params_from_dict, suggest_retrieval_params

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class Stage1Result:
    best_params: RetrievalParams
    best_value: float
    baseline_value: float
    metric: str
    n_trials: int
    total_seconds: float


class Stage1Optimizer:
    def __init__(self, cfg: Config, pairs: list[QAPair],
                 trial_log_path: Optional[str | Path] = None) -> None:
        self.cfg = cfg
        self.pairs = pairs
        stage1 = cfg.optimization.get("stage1_retrieval", {})
        self.metric = stage1.get("metric", "mrr")
        self.n_trials = int(stage1.get("n_trials", 30))
        self.seed = cfg.splits.seed
        self.trial_log_path = Path(trial_log_path) if trial_log_path else None

        self.docs = load_corpus(cfg.dataset.corpus_path, cfg.dataset.corpus_glob)
        self.embedder = build_embedder(cfg.models.embedding)
        self._reranker = None
        self._index_cache: dict[tuple[int, int], tuple[list[Chunk], HybridRetriever]] = {}
        self._n = 0  # trial counter for logging

    def _reranker_lazy(self):
        if self._reranker is None:
            self._reranker = build_reranker(self.cfg.models.reranker)
        return self._reranker

    def _get_index(self, chunk_size: int, chunk_overlap: int):
        key = (chunk_size, chunk_overlap)
        if key not in self._index_cache:
            chunks = chunk_documents(self.docs, chunk_size, chunk_overlap)
            embeddings = self.embedder.embed([c.text for c in chunks])
            hybrid = HybridRetriever(VectorStore(embeddings), BM25Index([c.text for c in chunks]))
            self._index_cache[key] = (chunks, hybrid)
        return self._index_cache[key]

    def _evaluate(self, params: RetrievalParams) -> dict[str, float]:
        chunks, hybrid = self._get_index(params.chunk_size, params.chunk_overlap)
        reranker = self._reranker_lazy() if params.use_reranker else None
        pipeline = RetrievePipeline(chunks, self.embedder, hybrid, reranker)
        samples = [
            ([r.chunk.doc_id for r in pipeline.retrieve(p.question, params)], p.gold_doc_id)
            for p in self.pairs
        ]
        return retrieval_metrics(samples)

    def _objective(self, trial: optuna.Trial) -> float:
        params = suggest_retrieval_params(trial, self.cfg.search_space)
        if params.chunk_overlap >= params.chunk_size:
            raise optuna.TrialPruned()  # invalid chunking; let Optuna skip it

        t0 = time.perf_counter()
        metrics = self._evaluate(params)
        elapsed = round(time.perf_counter() - t0, 4)

        trial.set_user_attr("time_sec", elapsed)
        trial.set_user_attr("metrics", metrics)
        self._n += 1
        if self.trial_log_path is not None:
            append_jsonl(self.trial_log_path, {
                "trial": self._n,
                "params": asdict(params),
                "metric": self.metric,
                "value": metrics[self.metric],
                "metrics": metrics,
                "time_sec": elapsed,      # per-config timing
            })
        return metrics[self.metric]

    def run(self) -> Stage1Result:
        baseline = self._evaluate(baseline_retrieval_params(self.cfg))[self.metric]
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.seed))
        t0 = time.perf_counter()
        study.optimize(self._objective, n_trials=self.n_trials)
        total = round(time.perf_counter() - t0, 4)

        best_params = params_from_dict(study.best_params)
        return Stage1Result(
            best_params=best_params,
            best_value=round(study.best_value, 4),
            baseline_value=round(baseline, 4),
            metric=self.metric,
            n_trials=len(study.trials),
            total_seconds=total,
        )
