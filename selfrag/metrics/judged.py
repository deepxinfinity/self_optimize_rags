"""Judged metrics — faithfulness and answer_relevancy via the LLM judge.

Expensive (an LLM call per metric per sample), so Stage 2 keeps sample sizes and trial counts
modest and leans on the judge's disk cache. Scores are in [0, 1].
"""

from __future__ import annotations

from statistics import mean
from typing import Any

DEFAULT_METRICS = ("faithfulness", "answer_relevancy")


class JudgedEvaluator:
    def __init__(self, judge: Any, metrics: tuple[str, ...] = DEFAULT_METRICS) -> None:
        self.judge = judge
        self.metrics = metrics

    def score_pair(self, question: str, answer: str, context: str) -> dict[str, float]:
        return {m: self.judge.score(m, question, answer, context) for m in self.metrics}

    def evaluate(self, samples: list[dict[str, str]]) -> dict[str, float]:
        """``samples``: list of {"question","answer","context"}. Returns mean per metric + n."""
        if not samples:
            return {m: 0.0 for m in self.metrics} | {"n": 0}
        per_metric: dict[str, list[float]] = {m: [] for m in self.metrics}
        for s in samples:
            scored = self.score_pair(s["question"], s["answer"], s["context"])
            for m, v in scored.items():
                per_metric[m].append(v)
        result = {m: round(mean(vals), 4) for m, vals in per_metric.items()}
        result["n"] = len(samples)
        return result
