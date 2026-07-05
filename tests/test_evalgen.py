"""Increment-2 tests for splits + eval-set IO. Fast: no model calls."""

from __future__ import annotations

import pytest

from selfrag.evalgen.generator import QAPair
from selfrag.evalgen.splits import load_jsonl, split_pairs, write_jsonl


def _pairs(n: int) -> list[QAPair]:
    return [
        QAPair(id=f"arm-{i:05d}", arm_id="arm", question=f"q{i}", answer=f"a{i}",
               gold_doc_id=f"doc{i % 3}.md", gold_chunk_id=f"doc{i % 3}.md::0",
               source_text=f"passage {i}")
        for i in range(n)
    ]


def test_split_ratios_and_disjoint():
    pairs = _pairs(100)
    s = split_pairs(pairs, 0.6, 0.2, 0.2, seed=42)
    assert len(s["train"]) == 60 and len(s["val"]) == 20 and len(s["test"]) == 20
    ids = [p.id for sub in s.values() for p in sub]
    assert len(ids) == len(set(ids)) == 100          # complete + disjoint


def test_split_deterministic():
    pairs = _pairs(50)
    a = split_pairs(pairs, 0.6, 0.2, 0.2, seed=7)
    b = split_pairs(pairs, 0.6, 0.2, 0.2, seed=7)
    assert [p.id for p in a["val"]] == [p.id for p in b["val"]]


def test_split_seed_changes_partition():
    pairs = _pairs(50)
    a = split_pairs(pairs, 0.6, 0.2, 0.2, seed=1)
    b = split_pairs(pairs, 0.6, 0.2, 0.2, seed=2)
    assert [p.id for p in a["val"]] != [p.id for p in b["val"]]


def test_split_rejects_bad_ratios():
    with pytest.raises(ValueError):
        split_pairs(_pairs(10), 0.5, 0.3, 0.3, seed=0)


def test_jsonl_roundtrip(tmp_path):
    pairs = _pairs(5)
    path = tmp_path / "arm" / "all.jsonl"
    write_jsonl(pairs, path)
    loaded = load_jsonl(path)
    assert [p.to_dict() for p in loaded] == [p.to_dict() for p in pairs]
