from __future__ import annotations

import json

import pytest

from selfrag.corpus.chunker import Chunk
from selfrag.evalgen.generator import EvalGenerator, QAPair, is_grounded
from selfrag.evalgen.splits import load_jsonl, split_pairs, write_jsonl

PASSAGE = ("The encoder has six layers. Each layer uses multi-head attention with eight heads, "
           "followed by a feed-forward network with an inner size of 2048 units.")


def _pairs(n: int) -> list[QAPair]:
    return [
        QAPair(id=f"arm-{i:05d}", arm_id="arm", question=f"q{i}", answer=f"a{i}",
               gold_doc_id=f"doc{i % 3}.md", gold_chunk_id=f"doc{i % 3}.md::0",
               source_text=f"passage {i}", evidence=[f"passage {i}"])
        for i in range(n)
    ]


def test_split_ratios_and_disjoint():
    s = split_pairs(_pairs(100), 0.6, 0.2, 0.2, seed=42)
    assert len(s["train"]) == 60 and len(s["val"]) == 20 and len(s["test"]) == 20
    ids = [p.id for sub in s.values() for p in sub]
    assert len(ids) == len(set(ids)) == 100


def test_split_deterministic():
    a = split_pairs(_pairs(50), 0.6, 0.2, 0.2, seed=7)
    b = split_pairs(_pairs(50), 0.6, 0.2, 0.2, seed=7)
    assert [p.id for p in a["val"]] == [p.id for p in b["val"]]


def test_split_seed_changes_partition():
    a = split_pairs(_pairs(50), 0.6, 0.2, 0.2, seed=1)
    b = split_pairs(_pairs(50), 0.6, 0.2, 0.2, seed=2)
    assert [p.id for p in a["val"]] != [p.id for p in b["val"]]


def test_split_rejects_bad_ratios():
    with pytest.raises(ValueError):
        split_pairs(_pairs(10), 0.5, 0.3, 0.3, seed=0)


def test_jsonl_roundtrip_keeps_evidence(tmp_path):
    pairs = _pairs(5)
    path = tmp_path / "arm" / "all.jsonl"
    write_jsonl(pairs, path)
    assert [p.to_dict() for p in load_jsonl(path)] == [p.to_dict() for p in pairs]


def test_old_eval_sets_without_evidence_still_load(tmp_path):
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps({"id": "x", "arm_id": "a", "question": "q", "answer": "a",
                                "gold_doc_id": "d", "gold_chunk_id": "d::0",
                                "source_text": "s"}) + "\n")
    assert load_jsonl(path)[0].evidence == []


def test_is_grounded():
    assert is_grounded("Each layer uses multi-head attention with eight heads", PASSAGE)
    assert not is_grounded("Each layer uses sparse attention with sixteen heads", PASSAGE)
    assert not is_grounded("", PASSAGE)


class _ScriptedLLM:
    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = iter(replies)
        self.calls = 0

    def generate(self, prompt, system=None, **_):
        self.calls += 1
        return next(self.replies, "not json")


def _generator(tmp_path, replies) -> EvalGenerator:
    gen = EvalGenerator({"id": "arm", "provider": "ollama", "name": "x"},
                        cache_dir=str(tmp_path), seed=1)
    gen.llm = _ScriptedLLM(replies)
    return gen


def _reply(q, a, ev):
    return json.dumps({"question": q, "answer": a, "evidence": ev})


def test_generator_drops_unparsed_ungrounded_and_duplicate_pairs(tmp_path):
    chunks = [Chunk(f"d.txt::{i}", "d.txt", f"{PASSAGE} Section {i}.", i) for i in range(5)]
    replies = [
        _reply("How many heads?", "eight", "multi-head attention with eight heads"),
        "not json at all",
        _reply("How big is the inner layer?", "4096", "an inner size of 4096 hidden units"),
        _reply("How many heads?", "eight", "multi-head attention with eight heads"),
        _reply("How many layers?", "six", "The encoder has six layers."),
    ]
    gen = _generator(tmp_path, replies)
    pairs = gen.generate(chunks, n_pairs=2)
    assert sorted(p.question for p in pairs) == ["How many heads?", "How many layers?"]
    assert all(len(p.evidence) == 1 for p in pairs)
    assert dict(gen.stats) == {"unparsed": 1, "ungrounded": 1, "duplicate": 1,
                               "attempts": 5, "kept": 2}


def test_generator_caches_failures_too(tmp_path):
    chunks = [Chunk("d.txt::0", "d.txt", PASSAGE, 0)]
    first = _generator(tmp_path, [])
    assert first.generate(chunks, n_pairs=1) == []
    again = _generator(tmp_path, [])
    assert again.generate(chunks, n_pairs=1) == []
    assert first.llm.calls == 3 and again.llm.calls == 0


def test_generate_arm_evalset_scopes_questions_with_the_document_title(tmp_path, monkeypatch):
    from selfrag.config import load_config
    from selfrag.corpus.loader import Document, document_title
    from selfrag.evalgen import generator

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "p1.txt").write_text(f"# Attention Is All You Need\n\n{PASSAGE}")
    cfg = load_config("configs/qasper.yaml", [f"dataset.corpus_path={corpus}",
                                              f"runtime.cache_dir={tmp_path / 'cache'}",
                                              "evalgen.n_pairs_per_arm=1"])
    llm = _ScriptedLLM([_reply("How many heads?", "eight", "multi-head attention with eight heads")])
    monkeypatch.setattr(generator, "build_llm", lambda *a, **k: llm)

    pairs, stats = generator.generate_arm_evalset(
        cfg, cfg.models.eval_generators[0], generator.build_eval_chunks(cfg))
    assert pairs[0].question == 'In the paper "Attention Is All You Need": How many heads?'
    assert stats["kept"] == 1
    assert document_title(Document("x.txt", "x.txt", "no heading here\nmore")) == "no heading here"


def test_generator_reports_progress(tmp_path):
    chunks = [Chunk(f"d.txt::{i}", "d.txt", f"{PASSAGE} Section {i}.", i) for i in range(6)]
    replies = [_reply(f"Question {i}?", "six", "The encoder has six layers.") for i in range(6)]
    lines: list[str] = []
    _generator(tmp_path, replies).generate(chunks, n_pairs=5, log=lines.append, log_every=2)
    assert lines[0].startswith("  arm: 1/5 pairs after 1 calls")
    assert any("min left" in line for line in lines)


def test_generator_counts_and_caches_aborted_replies(tmp_path):
    from selfrag.models.ollama_adapter import ModelOutputError

    class Loops(_ScriptedLLM):
        def generate(self, prompt, system=None, **_):
            self.calls += 1
            if self.calls == 1:
                raise ModelOutputError("gemma2:9b: prediction aborted")
            return _reply(f"Q{self.calls}?", "six", "The encoder has six layers.")

    chunks = [Chunk(f"d.txt::{i}", "d.txt", f"{PASSAGE} Section {i}.", i) for i in range(4)]
    gen = _generator(tmp_path, [])
    gen.llm = Loops([])
    assert len(gen.generate(chunks, n_pairs=2)) == 2
    assert gen.stats["aborted"] == 1 and gen.llm.calls == 3

    again = _generator(tmp_path, [])
    assert len(again.generate(chunks, n_pairs=2)) == 2
    assert again.llm.calls == 0 and again.stats["aborted"] == 1
