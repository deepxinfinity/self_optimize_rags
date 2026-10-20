from __future__ import annotations

import pytest

from selfrag.corpus.chunker import Chunk
from selfrag.metrics.answer import token_f1
from selfrag.metrics.judged import JudgedEvaluator
from selfrag.metrics.retrieval import (hit, passage_relevant, reciprocal_rank, retrieval_record,
                                       summarize_retrieval)
from selfrag.models.judge import LLMJudge, _normalize

EVIDENCE = ("The model was trained with Adam at a learning rate of 0.001 for ten epochs "
            "on the combined training data.")


def _chunk(doc: str, i: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"{doc}::{i}", doc_id=doc, text=text, ordinal=i)


def test_hit_and_rr():
    assert hit(["a", "b", "c"], "b") is True
    assert hit(["a", "c"], "b") is False
    assert reciprocal_rank(["x", "gold", "y"], "gold") == 0.5
    assert reciprocal_rank(["gold", "y"], "gold") == 1.0
    assert reciprocal_rank(["x", "y"], "gold") == 0.0


def test_passage_relevant_when_evidence_is_inside_a_bigger_chunk():
    chunk = "Some setup text about the corpus. " + EVIDENCE + " Results follow in Table 2."
    assert passage_relevant(chunk, [EVIDENCE])


def test_passage_relevant_when_a_small_chunk_sits_inside_long_evidence():
    long_evidence = " ".join(f"w{i}" for i in range(300))
    small_chunk = " ".join(f"w{i}" for i in range(100, 160))
    assert passage_relevant(small_chunk, [long_evidence])


def test_passage_not_relevant_for_unrelated_text():
    chunk = "We evaluate on three benchmarks and report accuracy for each language pair."
    assert not passage_relevant(chunk, [EVIDENCE])


def test_passage_relevant_needs_half_the_ngrams():
    words = EVIDENCE.split()
    first_third = " ".join(words[: len(words) // 3])
    assert not passage_relevant(first_third + " unrelated filler words here and there", [EVIDENCE])


def test_retrieval_record_doc_and_passage_rank():
    chunks = [_chunk("other.txt", 0, "nothing useful here at all"),
              _chunk("gold.txt", 3, "intro paragraph of the gold paper without the answer"),
              _chunk("gold.txt", 7, EVIDENCE)]
    rec = retrieval_record(chunks, "gold.txt", [EVIDENCE])
    assert rec["hit"] and rec["rr"] == 0.5
    assert rec["passage_hit"] and rec["passage_rr"] == pytest.approx(1 / 3)
    assert rec["retrieved"] == ["other.txt::0", "gold.txt::3", "gold.txt::7"]


def test_retrieval_record_ignores_evidence_text_from_other_docs():
    chunks = [_chunk("copy.txt", 0, EVIDENCE)]
    rec = retrieval_record(chunks, "gold.txt", [EVIDENCE])
    assert rec["passage_hit"] is False


def test_retrieval_record_without_evidence_has_no_passage_fields():
    rec = retrieval_record([_chunk("gold.txt", 0, "x")], "gold.txt")
    assert "passage_hit" not in rec


def test_summarize_retrieval_counts_passage_metrics_only_where_evidence_exists():
    records = [
        {"hit": True, "rr": 1.0, "passage_hit": True, "passage_rr": 0.5},
        {"hit": True, "rr": 0.5, "passage_hit": False, "passage_rr": 0.0},
        {"hit": False, "rr": 0.0},
    ]
    m = summarize_retrieval(records)
    assert m["n"] == 3
    assert m["hit_rate"] == round(2 / 3, 4)
    assert m["mrr"] == 0.5
    assert m["n_with_evidence"] == 2
    assert m["passage_hit_rate"] == 0.5
    assert m["passage_mrr"] == 0.25


def test_summarize_retrieval_empty():
    assert summarize_retrieval([]) == {"hit_rate": 0.0, "mrr": 0.0, "n": 0}


@pytest.mark.parametrize("pred,ref,expected", [
    ("Adam", "Adam", 1.0),
    ("The Adam optimizer", "adam optimizer", 1.0),
    ("SGD", "Adam", 0.0),
    ("they used Adam", "Adam", 0.5),
    ("", "Adam", 0.0),
])
def test_token_f1(pred, ref, expected):
    assert token_f1(pred, ref) == pytest.approx(expected)


def test_normalize_scale():
    assert _normalize(5) == 1.0
    assert _normalize(1) == 0.0
    assert _normalize(3) == 0.5
    assert _normalize(0) == 0.0
    assert _normalize(9) == 1.0


class _RecordingLLM:
    def __init__(self, reply: str = '{"score": 4, "reason": "ok"}') -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt, system=None, **_):
        self.prompts.append(prompt)
        return self.reply


def test_judge_sends_only_the_fields_each_metric_needs(tmp_path):
    llm = _RecordingLLM()
    judge = LLMJudge(llm, "stub", cache_dir=str(tmp_path))
    judge.score("faithfulness", "q?", "a", context="CTX", reference="REF")
    judge.score("answer_relevancy", "q?", "a", context="CTX", reference="REF")
    judge.score("answer_correctness", "q?", "a", context="CTX", reference="REF")
    faith, relevancy, correctness = llm.prompts
    assert "CTX" in faith and "REF" not in faith
    assert "CTX" not in relevancy and "REF" not in relevancy
    assert "REF" in correctness and "CTX" not in correctness


def test_judge_correctness_requires_reference(tmp_path):
    judge = LLMJudge(_RecordingLLM(), "stub", cache_dir=str(tmp_path))
    with pytest.raises(ValueError):
        judge.score("answer_correctness", "q?", "a")


def test_judge_caches_scores(tmp_path):
    llm = _RecordingLLM()
    judge = LLMJudge(llm, "stub", cache_dir=str(tmp_path))
    assert judge.score("answer_relevancy", "q?", "a") == 0.75
    assert judge.score("answer_relevancy", "q?", "a") == 0.75
    assert len(llm.prompts) == 1


class _StubJudge:
    def score(self, metric, question, answer, context="", reference=""):
        return {"faithfulness": 1.0, "answer_relevancy": 0.5, "answer_correctness": 0.25}[metric]


def test_judged_evaluator_aggregates():
    ev = JudgedEvaluator(_StubJudge())
    samples = [{"question": "q", "answer": "adam", "context": "c", "reference": "adam"}] * 4
    summary, items = ev.evaluate(samples)
    assert summary["n"] == 4
    assert summary["faithfulness"] == 1.0 and summary["answer_relevancy"] == 0.5
    assert summary["answer_correctness"] == 0.25 and summary["answer_f1"] == 1.0
    assert len(items) == 4 and items[0]["answer_f1"] == 1.0


def test_judged_evaluator_skips_correctness_without_reference():
    ev = JudgedEvaluator(_StubJudge())
    summary, items = ev.evaluate([{"question": "q", "answer": "a", "context": "c"}])
    assert items[0]["answer_correctness"] is None
    assert "answer_correctness" not in summary and "answer_f1" not in summary
    assert summary["faithfulness"] == 1.0
