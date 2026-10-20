from __future__ import annotations

import json

import pytest

from conftest import CountingEmbedder, LengthReranker

from selfrag.config import load_config
from selfrag.evalgen.generator import QAPair
from selfrag.evalgen.heldout import HeldoutAccessError, HeldoutSet, load_real_questions
from selfrag.evalgen.splits import write_jsonl
from selfrag.optimize import runner as runner_mod
from selfrag.optimize.runner import (format_reeval, format_significance, format_table,
                                     run_optimize, run_reeval)
from selfrag.rag import RagFactory


def _pairs(n=3):
    return [QAPair(id=str(i), arm_id="a", question=f"q{i}", answer="x", gold_doc_id="d.md")
            for i in range(n)]


def test_heldout_locked_by_default():
    with pytest.raises(HeldoutAccessError):
        HeldoutSet("test", _pairs()).take(by="optimizer")


def test_heldout_unlock_allows_access_and_records():
    h = HeldoutSet("test", _pairs())
    h.unlock()
    assert len(h.take(by="final_eval")) == 3
    h.take(by="baseline")
    assert h.accesses == ["final_eval", "baseline"]


def test_load_real_questions_reads_evidence(tmp_path):
    p = tmp_path / "real.jsonl"
    p.write_text(json.dumps({"question": "why?", "answer": "because", "gold_doc_id": "d.md",
                             "evidence": ["because it was"]}) + "\n")
    pairs = load_real_questions(p)
    assert pairs[0].arm_id == "real" and pairs[0].evidence == ["because it was"]


def test_load_real_questions_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_real_questions(tmp_path / "nope.jsonl")


def test_format_table_shapes_rows():
    report = {
        "headline_metric": "answer_correctness",
        "baseline": {"scores": {"synthetic_test_mean": 0.60, "real_heldout": 0.58}},
        "arms": {
            "local_qg": {"scores": {"synthetic_val": 0.81, "synthetic_test": 0.78, "real_heldout": 0.72}},
            "frontier": {"scores": {"synthetic_val": 0.83, "synthetic_test": 0.79, "real_heldout": None}},
        },
    }
    table = format_table(report)
    assert "Real held-out" in table
    assert "Optimized — local_qg" in table and "Optimized — frontier" in table
    assert "0.720" in table and "0.580" in table and "—" in table


class _FakeLLM:
    name = "fake"

    def generate(self, prompt, system=None, **_):
        if "CURRENT PROMPT" in prompt:
            return json.dumps({"prompt": f"{system[:10]} improved {len(prompt)}"})
        # answer with the first words of the first retrieved chunk
        return " ".join(prompt.split("\n")[2].split()[:3])


class _FakeJudge:
    def score(self, metric, question, answer, context="", reference="", **_):
        return 1.0 if reference and reference.split()[0] in answer else 0.25


class _FakeFactory(RagFactory):
    def __init__(self, cfg, timing=None):
        super().__init__(cfg, timing)
        self._embedder = CountingEmbedder()
        self._reranker = LengthReranker()
        self._generator = _FakeLLM()


def _write_study(tmp_path, corpus):
    out = tmp_path / "evalsets"
    for arm in ("local_qg", "frontier"):
        for split, n in (("train", 6), ("val", 3), ("test", 3)):
            pairs = []
            for i in range(n):
                doc = i % 3
                words = (corpus / f"doc{doc}.txt").read_text().split()
                ev = " ".join(words[50 * i:50 * i + 12])
                pairs.append(QAPair(id=f"{arm}-{split}-{i}", arm_id=arm, question=ev,
                                    answer=words[50 * i], gold_doc_id=f"doc{doc}.txt",
                                    evidence=[ev]))
            write_jsonl(pairs, out / arm / f"{split}.jsonl")
    real = tmp_path / "real.jsonl"
    with real.open("w") as f:
        for i in range(4):
            words = (corpus / f"doc{i % 3}.txt").read_text().split()
            ev = " ".join(words[300 + i:312 + i])
            f.write(json.dumps({"question": ev, "answer": words[300 + i],
                                "gold_doc_id": f"doc{i % 3}.txt", "evidence": [ev]}) + "\n")
    return out, real


def _study_config(tmp_path, corpus, out, real):
    return load_config("configs/qasper.yaml", [
        f"dataset.corpus_path={corpus}",
        f"dataset.real_questions_path={real}",
        f"evalgen.out_dir={out}",
        f"runtime.cache_dir={tmp_path / 'cache'}",
        f"runtime.embedding_cache_dir={tmp_path / 'emb'}",
        "optimization.stage1_retrieval.n_trials=4",
        "optimization.stage2_prompts.n_trials=2",
    ])


def _use_fakes(monkeypatch):
    monkeypatch.setattr(runner_mod, "RagFactory", _FakeFactory)
    monkeypatch.setattr(runner_mod, "build_judge", lambda *a, **k: _FakeJudge())
    monkeypatch.setattr("selfrag.optimize.stage2_prompts.build_llm", lambda *a, **k: _FakeLLM())


def test_run_optimize_end_to_end_with_fakes(tmp_path, corpus, monkeypatch):
    out, real = _write_study(tmp_path, corpus)
    cfg = _study_config(tmp_path, corpus, out, real)
    _use_fakes(monkeypatch)

    run_dir = tmp_path / "run"
    report = run_optimize(cfg, run_dir, log=lambda msg: None)

    assert set(report["arms"]) == {"local_qg", "frontier"}
    for arm in report["arms"].values():
        assert arm["stage1"]["best"] >= arm["stage1"]["baseline"]
        assert arm["scores"]["real_heldout"] is not None
        assert "passage_hit_rate" in arm["scores"]["real_heldout_full"]["retrieval"]
    assert report["baseline"]["scores"]["real_heldout"] is not None
    assert set(report["significance"]["real_heldout"]) == {
        "local_qg_vs_baseline", "frontier_vs_baseline", "local_qg_vs_frontier"}
    assert report["heldout_access_log"]["real"] == [
        "optimized:local_qg", "optimized:frontier", "baseline"]

    saved = json.loads((run_dir / "best_configs.json").read_text())
    assert saved["headline_metric"] == "answer_correctness"
    assert (run_dir / "per_question" / "baseline__real_heldout__judged.jsonl").exists()
    assert (run_dir / "local_qg_stage1_trials.jsonl").exists()
    assert (run_dir / "timings.json").exists()

    assert "Optimized — frontier" in format_table(report)
    assert "local_qg vs frontier" in format_significance(report)


def test_reeval_scores_the_chosen_configs_on_another_question_file(tmp_path, corpus, monkeypatch):
    out, real = _write_study(tmp_path, corpus)
    _use_fakes(monkeypatch)
    run_dir = tmp_path / "run"
    chosen = run_optimize(_study_config(tmp_path, corpus, out, real), run_dir,
                          log=lambda msg: None)

    other = tmp_path / "other.jsonl"
    rows = [json.loads(line) for line in real.read_text().splitlines()]
    other.write_text("".join(json.dumps({**r, "question": r["question"].split()[0]}) + "\n"
                             for r in rows))
    cfg = _study_config(tmp_path, corpus, out, other)
    report = run_reeval(cfg, run_dir, tmp_path / "reeval", log=lambda msg: None)

    assert list(report["configs"]) == ["baseline", "local_qg", "frontier"]
    assert (report["configs"]["local_qg"]["retrieval"]
            == chosen["arms"]["local_qg"]["best_config"]["retrieval"])
    assert report["configs"]["baseline"]["scores"]["retrieval"]["n"] == len(rows)
    assert set(report["significance"]["real_heldout"]) == {
        "local_qg_vs_baseline", "frontier_vs_baseline", "local_qg_vs_frontier"}
    assert (tmp_path / "reeval" / "reeval.json").exists()
    assert (tmp_path / "reeval" / "per_question" / "frontier__real_heldout__judged.jsonl").exists()
    text = format_reeval(report)
    assert "Optimized — frontier" in text and str(other) in text
    assert "local_qg vs frontier" in format_significance(report)


def test_run_optimize_reports_progress(tmp_path, corpus, monkeypatch):
    out, real = _write_study(tmp_path, corpus)
    _use_fakes(monkeypatch)
    lines: list[str] = []
    run_optimize(_study_config(tmp_path, corpus, out, real), tmp_path / "run", log=lines.append)
    text = "\n".join(lines)
    assert "trial  0/4: passage_hit_rate=" in text
    assert "seed prompt: answer_correctness=" in text
    assert "final eval: baseline__real_heldout (retrieval on 4, judging 4)" in text
