"""Increment-3 metric tests. Fast: pure functions + a stub judge, no model calls."""

from __future__ import annotations

from selfrag.metrics.judged import JudgedEvaluator
from selfrag.metrics.retrieval import hit, reciprocal_rank, retrieval_metrics
from selfrag.models.judge import _normalize


def test_hit_and_rr():
    assert hit(["a", "b", "c"], "b") is True
    assert hit(["a", "c"], "b") is False
    assert reciprocal_rank(["x", "gold", "y"], "gold") == 0.5
    assert reciprocal_rank(["gold", "y"], "gold") == 1.0
    assert reciprocal_rank(["x", "y"], "gold") == 0.0


def test_retrieval_metrics_aggregate():
    samples = [
        (["gold", "x"], "gold"),     # hit, rr=1.0
        (["x", "gold"], "gold"),     # hit, rr=0.5
        (["x", "y"], "gold"),        # miss, rr=0.0
    ]
    m = retrieval_metrics(samples)
    assert m["n"] == 3
    assert m["hit_rate"] == round(2 / 3, 4)
    assert m["mrr"] == round((1.0 + 0.5 + 0.0) / 3, 4)


def test_retrieval_metrics_empty():
    assert retrieval_metrics([]) == {"hit_rate": 0.0, "mrr": 0.0, "n": 0}


def test_normalize_scale():
    assert _normalize(5) == 1.0        # top of 1-5 rubric
    assert _normalize(1) == 0.0        # bottom
    assert _normalize(3) == 0.5        # middle
    assert _normalize(0) == 0.0        # clamps below 1
    assert _normalize(9) == 1.0        # clamps above 5


class _StubJudge:
    """Returns a fixed score per metric — lets us test aggregation without an LLM."""
    def score(self, metric, question, answer, context):
        return {"faithfulness": 1.0, "answer_relevancy": 0.5}[metric]


def test_judged_evaluator_aggregates():
    ev = JudgedEvaluator(_StubJudge())
    samples = [{"question": "q", "answer": "a", "context": "c"} for _ in range(4)]
    res = ev.evaluate(samples)
    assert res["faithfulness"] == 1.0 and res["answer_relevancy"] == 0.5 and res["n"] == 4
