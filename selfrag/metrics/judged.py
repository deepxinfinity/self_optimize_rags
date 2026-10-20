from __future__ import annotations

from statistics import mean
from typing import Any, Callable, Optional

from .answer import token_f1

DEFAULT_METRICS = ("answer_correctness", "faithfulness", "answer_relevancy")


class JudgedEvaluator:
    def __init__(self, judge: Any, metrics: tuple[str, ...] = DEFAULT_METRICS) -> None:
        self.judge = judge
        self.metrics = metrics

    def score_one(self, sample: dict[str, str]) -> dict[str, Optional[float]]:
        reference = sample.get("reference") or ""
        scores: dict[str, Optional[float]] = {}
        for m in self.metrics:
            if m == "answer_correctness" and not reference:
                scores[m] = None
                continue
            scores[m] = self.judge.score(m, sample["question"], sample["answer"],
                                         context=sample.get("context", ""), reference=reference)
        if reference:
            scores["answer_f1"] = token_f1(sample["answer"], reference)
        return scores

    def evaluate(self, samples: list[dict[str, str]],
                 log: Optional[Callable[[str], None]] = None) -> tuple[dict[str, Any], list[dict]]:
        """Samples are {question, answer, context, reference?}; returns (means, per-item scores)."""
        items = []
        for i, s in enumerate(samples, start=1):
            items.append({"question": s["question"], "answer": s["answer"], **self.score_one(s)})
            if log and i % 25 == 0:
                log(f"    judged {i}/{len(samples)}")
        return summarize_judged(items, self.metrics), items


def summarize_judged(items: list[dict[str, Any]], metrics: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(items)}
    for m in (*metrics, "answer_f1"):
        vals = [it[m] for it in items if it.get(m) is not None]
        if vals:
            out[m] = round(mean(vals), 4)
        if m != "answer_correctness" and 0 < len(vals) < len(items):
            out[f"{m}_n"] = len(vals)   # some judge replies were cut off
    return out
