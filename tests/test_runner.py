"""Increment-6 tests: held-out access guard, real-questions loader, table formatting."""

from __future__ import annotations

import json

import pytest

from selfrag.evalgen.generator import QAPair
from selfrag.evalgen.heldout import HeldoutAccessError, HeldoutSet, load_real_questions
from selfrag.optimize.runner import format_table


def _pairs(n=3):
    return [QAPair(id=str(i), arm_id="a", question=f"q{i}", answer="x",
                   gold_doc_id="d.md", gold_chunk_id="", source_text="") for i in range(n)]


def test_heldout_locked_by_default():
    h = HeldoutSet("test", _pairs())
    with pytest.raises(HeldoutAccessError):
        h.take(by="optimizer")           # accessing before unlock must fail loudly


def test_heldout_unlock_allows_access_and_records():
    h = HeldoutSet("test", _pairs())
    h.unlock()
    got = h.take(by="final_eval")
    assert len(got) == 3
    h.take(by="baseline")
    assert h.accesses == ["final_eval", "baseline"]   # audit trail for the run log


def test_load_real_questions(tmp_path):
    p = tmp_path / "real.jsonl"
    p.write_text(
        json.dumps({"question": "why?", "answer": "because", "gold_doc_id": "d.md"}) + "\n"
    )
    pairs = load_real_questions(p)
    assert len(pairs) == 1
    assert pairs[0].arm_id == "real" and pairs[0].gold_doc_id == "d.md"


def test_load_real_questions_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_real_questions(tmp_path / "nope.jsonl")


def test_format_table_shapes_rows():
    report = {
        "headline_metric": "faithfulness",
        "baseline": {"scores": {"synthetic_test_mean": 0.60, "real_heldout": 0.58}},
        "arms": {
            "local_qg": {"scores": {"synthetic_val": 0.81, "synthetic_test": 0.78, "real_heldout": 0.72}},
            "frontier": {"scores": {"synthetic_val": 0.83, "synthetic_test": 0.79, "real_heldout": 0.74}},
        },
    }
    table = format_table(report)
    assert "Real held-out" in table
    assert "Optimized — local_qg" in table and "Optimized — frontier" in table
    assert "0.720" in table and "0.580" in table   # values rendered
    assert "—" in table                             # baseline has no val score
