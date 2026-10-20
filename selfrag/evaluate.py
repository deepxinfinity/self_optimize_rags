from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Optional

from .evalgen.generator import QAPair
from .metrics.judged import DEFAULT_METRICS, JudgedEvaluator
from .metrics.retrieval import retrieval_record, summarize_retrieval
from .rag import RAG
from .retrieve.pipeline import RetrievalResult
from .utils.timing import TimingLog, maybe_timed

Retrieved = list[list[RetrievalResult]]


def _retrieve_all(rag: RAG, pairs: list[QAPair], timing: Optional[TimingLog]) -> Retrieved:
    return [rag.pipeline.retrieve(p.question, rag.params, timing) for p in pairs]


def evaluate_retrieval(rag: RAG, pairs: list[QAPair], timing: Optional[TimingLog] = None,
                       retrieved: Optional[Retrieved] = None) -> tuple[dict, list[dict]]:
    retrieved = retrieved if retrieved is not None else _retrieve_all(rag, pairs, timing)
    records = [
        {"id": p.id, "question": p.question,
         **retrieval_record([r.chunk for r in results], p.gold_doc_id, p.evidence)}
        for p, results in zip(pairs, retrieved)
    ]
    return summarize_retrieval(records), records


def evaluate_answers(rag: RAG, judge: Any, pairs: list[QAPair],
                     metrics: tuple[str, ...] = DEFAULT_METRICS,
                     timing: Optional[TimingLog] = None,
                     retrieved: Optional[Retrieved] = None,
                     log: Optional[Callable[[str], None]] = None) -> tuple[dict, list[dict]]:
    # All answers are generated before any judging so ollama swaps between the generator
    # and judge models twice per call instead of twice per question.
    retrieved = retrieved if retrieved is not None else _retrieve_all(rag, pairs, timing)
    samples = []
    for i, (p, results) in enumerate(zip(pairs, retrieved), start=1):
        if log and i % 25 == 0:
            log(f"    answered {i}/{len(pairs)}")
        samples.append({
            "question": p.question,
            "answer": rag.answerer.answer(p.question, results, timing),
            "context": rag.answerer.format_context(results),
            "reference": p.answer,
        })
    with maybe_timed(timing, "judge", n=len(samples)):
        summary, items = JudgedEvaluator(judge, metrics).evaluate(samples, log=log)
    return summary, [{"id": p.id, **item} for p, item in zip(pairs, items)]


def evaluate(rag: RAG, pairs: list[QAPair], judge: Any = None,
             metrics: tuple[str, ...] = DEFAULT_METRICS, sample_size: Optional[int] = None,
             timing: Optional[TimingLog] = None,
             log: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
    """Retrieval metrics on every pair; answer metrics on the first `sample_size` if judged."""
    retrieved = _retrieve_all(rag, pairs, timing)
    retrieval, retrieval_items = evaluate_retrieval(rag, pairs, retrieved=retrieved)
    report: dict[str, Any] = {
        "n_pairs": len(pairs),
        "params": asdict(rag.params),
        "retrieval": retrieval,
        "per_question": {"retrieval": retrieval_items},
    }
    if judge is not None:
        n = min(sample_size, len(pairs)) if sample_size else len(pairs)
        report["judged"], report["per_question"]["judged"] = evaluate_answers(
            rag, judge, pairs[:n], metrics, timing, retrieved=retrieved[:n], log=log)
    return report
