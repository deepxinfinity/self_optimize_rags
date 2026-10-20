from __future__ import annotations

import json

import pytest

from selfrag.corpus.qasper import ingest_qasper

LONG = " ".join(["filler"] * 250)


def _answer(text=None, evidence=(), unanswerable=False, yes_no=None):
    return {"answer": {"unanswerable": unanswerable, "extractive_spans": [],
                       "yes_no": yes_no, "free_form_answer": text or "",
                       "evidence": list(evidence), "highlighted_evidence": []}}


def _paper(title, qas):
    return {"title": title, "abstract": "An abstract.",
            "full_text": [{"section_name": "Intro", "paragraphs": [LONG, "We use Adam."]}],
            "qas": [{"question": q, "answers": answers} for q, answers in qas]}


def _source(tmp_path):
    data = {
        "p1": _paper("Paper One", [
            ("What optimizer do they use?", [_answer("Adam", ["We use Adam."])]),
            ("Is it good?", [_answer(yes_no=True, evidence=["FLOAT SELECTED: Table 1"])]),
            ("Unanswerable?", [_answer(unanswerable=True)]),
        ]),
        "p2": _paper("Paper Two", [
            ("Which dataset?", [_answer("SQuAD"), _answer("SQuAD v1", ["We use Adam."])]),
        ]),
        "p3": _paper("No Questions", []),
    }
    src = tmp_path / "qasper.json"
    src.write_text(json.dumps(data))
    return src


def _ingest(tmp_path, **kw):
    out = tmp_path / "real.jsonl"
    summary = ingest_qasper(_source(tmp_path), tmp_path / "corpus", out, n_papers=10,
                            n_questions=10, **kw)
    return summary, [json.loads(line) for line in out.read_text().splitlines()]


def test_corpus_has_papers_with_answerable_questions(tmp_path):
    summary, rows = _ingest(tmp_path)
    assert summary["papers"] == 2
    assert sorted(p.name for p in (tmp_path / "corpus").iterdir()) == ["p1.txt", "p2.txt"]
    assert len(rows) == 3


def test_table_evidence_is_dropped_and_evidenced_answers_preferred(tmp_path):
    _, rows = _ingest(tmp_path)
    by_q = {r["question"]: r for r in rows}
    assert by_q["What optimizer do they use?"]["evidence"] == ["We use Adam."]
    assert by_q["Is it good?"]["evidence"] == []
    assert by_q["Which dataset?"]["answer"] == "SQuAD v1"


def test_require_evidence_filters_questions_not_papers(tmp_path):
    summary, rows = _ingest(tmp_path, require_evidence=True)
    assert summary["papers"] == 2
    assert all(r["evidence"] for r in rows) and len(rows) == 2


def test_scope_with_title(tmp_path):
    _, rows = _ingest(tmp_path, scope_with_title=True)
    row = next(r for r in rows if r["raw_question"] == "What optimizer do they use?")
    assert row["question"] == 'In the paper "Paper One": What optimizer do they use?'


def test_unscoped_copy_has_the_same_questions_in_the_same_order(tmp_path):
    unscoped = tmp_path / "unscoped.jsonl"
    _, scoped = _ingest(tmp_path, scope_with_title=True, unscoped_out=unscoped)
    raw = [json.loads(line) for line in unscoped.read_text().splitlines()]
    assert [r["question"] for r in raw] == [s["raw_question"] for s in scoped]
    assert [r["answer"] for r in raw] == [s["answer"] for s in scoped]
    assert [r["evidence"] for r in raw] == [s["evidence"] for s in scoped]
    assert all("raw_question" not in r for r in raw)


def test_unscoped_copy_needs_scoping(tmp_path):
    with pytest.raises(ValueError):
        _ingest(tmp_path, unscoped_out=tmp_path / "u.jsonl")
