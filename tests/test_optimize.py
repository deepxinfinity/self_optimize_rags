from __future__ import annotations

import json

import optuna
import pytest

from conftest import make_factory

from selfrag.config import load_config
from selfrag.evalgen.generator import QAPair
from selfrag.optimize.search_space import suggest_retrieval_params
from selfrag.optimize.stage1_retrieval import Stage1Optimizer
from selfrag.params import RetrievalParams, baseline_retrieval_params, params_from_dict

CONFIG = "configs/qasper.yaml"


def test_params_from_dict_types():
    p = params_from_dict({
        "chunk_size": "512", "chunk_overlap": "100", "top_k": "5",
        "hybrid_alpha": "0.3", "use_reranker": True, "rerank_candidates": "20",
    })
    assert isinstance(p, RetrievalParams)
    assert p.chunk_size == 512 and p.top_k == 5 and p.hybrid_alpha == 0.3
    assert p.rerank_candidates == 20 and p.query_rewrite_enabled is False


def test_params_from_dict_reads_old_runs():
    p = params_from_dict({"chunk_size": 256, "chunk_overlap": 50, "top_k": 6,
                          "hybrid_alpha": 0.3, "use_reranker": True, "rerank_top_k": 5})
    assert p.rerank_candidates == 5


def test_baseline_comes_from_config_section():
    cfg = load_config(CONFIG)
    p = baseline_retrieval_params(cfg)
    assert (p.chunk_size, p.chunk_overlap, p.top_k) == (256, 32, 5)
    assert p.use_reranker is False


def test_baseline_rejects_unknown_keys():
    cfg = load_config(CONFIG, ["baseline.chunk_sise=256"])
    with pytest.raises(ValueError):
        baseline_retrieval_params(cfg)


def test_suggest_respects_config_ranges():
    ss = load_config(CONFIG).search_space
    fixed = {"chunk_size": 384, "chunk_overlap": 64, "top_k": 7,
             "hybrid_alpha": 0.42, "use_reranker": False, "rerank_candidates": 40}
    p = suggest_retrieval_params(optuna.trial.FixedTrial(fixed), ss)
    assert p.chunk_size in ss["chunk_size"]["values"]
    assert p.chunk_overlap in ss["chunk_overlap"]["values"]
    assert ss["top_k"]["low"] <= p.top_k <= ss["top_k"]["high"]
    assert ss["hybrid_alpha"]["low"] <= p.hybrid_alpha <= ss["hybrid_alpha"]["high"]
    assert p.rerank_candidates in ss["rerank_candidates"]["values"]


def test_suggest_unknown_type_raises():
    with pytest.raises(ValueError):
        suggest_retrieval_params(optuna.trial.FixedTrial({"chunk_size": 256}),
                                 {"chunk_size": {"type": "bogus"}})


def _val_pairs(corpus) -> list[QAPair]:
    pairs = []
    for i in range(3):
        words = (corpus / f"doc{i}.txt").read_text().split()
        evidence = " ".join(words[200:215])
        pairs.append(QAPair(id=f"v{i}", arm_id="a", question=evidence, answer="x",
                            gold_doc_id=f"doc{i}.txt", evidence=[evidence]))
    return pairs


def test_stage1_scores_baseline_first_and_never_returns_worse(corpus, tmp_path):
    f = make_factory(corpus, tmp_path, "optimization.stage1_retrieval.n_trials=5")
    log = tmp_path / "trials.jsonl"
    result = Stage1Optimizer(f.cfg, _val_pairs(corpus), trial_log_path=log, factory=f).run()

    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[0]["label"] == "baseline" and rows[0]["trial"] == 0
    assert rows[0]["params"]["chunk_size"] == 256
    assert result.best_value >= result.baseline_value
    assert result.best_value == max(r["value"] for r in rows)
    assert result.metric == "passage_hit_rate"
    assert "passage_mrr" in result.best_metrics


def test_stage1_shares_the_factory_indices(corpus, tmp_path):
    f = make_factory(corpus, tmp_path, "optimization.stage1_retrieval.n_trials=8")
    Stage1Optimizer(f.cfg, _val_pairs(corpus), factory=f).run()
    assert f.timing.count("embed") == f.n_indices_built <= 9


def test_stage1_passage_metric_needs_evidence(corpus, tmp_path):
    f = make_factory(corpus, tmp_path)
    pairs = [QAPair(id="v0", arm_id="a", question="q", answer="x", gold_doc_id="doc0.txt")]
    with pytest.raises(ValueError, match="evidence"):
        Stage1Optimizer(f.cfg, pairs, factory=f)


def test_stage1_rank_prefers_cheaper_config_on_ties():
    base = {"value": 0.8, "metrics": {"passage_mrr": 0.5, "mrr": 0.6}}
    cheap = {**base, "params": {"use_reranker": False, "top_k": 4}}
    pricey = {**base, "params": {"use_reranker": True, "top_k": 8}}
    assert max([pricey, cheap], key=Stage1Optimizer._rank) is cheap
