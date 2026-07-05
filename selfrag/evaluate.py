"""Evaluation harness: score a built RAG against a set of QA pairs.

Two tiers, matching the two optimization stages:
  - retrieval metrics (cheap, no generation) -> Stage 1
  - judged metrics (expensive, LLM judge)    -> Stage 2

Both reuse the same RAG pipeline so a config is scored exactly as it would run in production.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from .evalgen.generator import QAPair
from .metrics.judged import JudgedEvaluator
from .metrics.retrieval import retrieval_metrics
from .rag import RAG
from .utils.timing import TimingLog


def evaluate_retrieval(rag: RAG, pairs: list[QAPair],
                       timing: Optional[TimingLog] = None) -> dict[str, float]:
    """hit_rate / MRR over the pairs. No answer generation."""
    samples: list[tuple[list[str], str]] = []
    for p in pairs:
        results = rag.pipeline.retrieve(p.question, rag.params, timing)
        samples.append(([r.chunk.doc_id for r in results], p.gold_doc_id))
    return retrieval_metrics(samples)


def evaluate_judged(rag: RAG, judge, pairs: list[QAPair], sample_size: Optional[int] = None,
                    timing: Optional[TimingLog] = None) -> dict[str, float]:
    """faithfulness / answer_relevancy over up to ``sample_size`` pairs (generation + judging)."""
    evaluator = JudgedEvaluator(judge)
    subset = pairs[:sample_size] if sample_size else pairs
    samples = []
    for p in subset:
        results = rag.pipeline.retrieve(p.question, rag.params, timing)
        answer = rag.answerer.answer(p.question, results, timing)
        context = rag.answerer.format_context(results)
        samples.append({"question": p.question, "answer": answer, "context": context})
    return evaluator.evaluate(samples)


def evaluate(rag: RAG, pairs: list[QAPair], judge=None, sample_size: Optional[int] = None,
             timing: Optional[TimingLog] = None) -> dict:
    """Full evaluation report: retrieval always; judged only if a judge is supplied."""
    report: dict = {
        "n_pairs": len(pairs),
        "params": asdict(rag.params),
        "retrieval": evaluate_retrieval(rag, pairs, timing),
    }
    if judge is not None:
        report["judged"] = evaluate_judged(rag, judge, pairs, sample_size, timing)
    return report
