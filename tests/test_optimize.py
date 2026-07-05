"""Increment-4 tests for the search space. Fast: Optuna FixedTrial, no models."""

from __future__ import annotations

import optuna
import pytest

from selfrag.config import load_config
from selfrag.optimize.search_space import params_from_dict, suggest_retrieval_params
from selfrag.params import RetrievalParams

CONFIG = "configs/default.yaml"


def test_params_from_dict_types():
    p = params_from_dict({
        "chunk_size": "512", "chunk_overlap": "100", "top_k": "5",
        "hybrid_alpha": "0.3", "use_reranker": True, "rerank_top_k": "3",
    })
    assert isinstance(p, RetrievalParams)
    assert p.chunk_size == 512 and p.top_k == 5 and p.hybrid_alpha == 0.3
    assert p.use_reranker is True and p.query_rewrite_enabled is False


def test_suggest_respects_config_ranges():
    cfg = load_config(CONFIG)
    ss = cfg.search_space
    # A FixedTrial returns the value we pin per param; assert each stays within the configured set.
    fixed = {
        "chunk_size": 1024, "chunk_overlap": 200, "top_k": 7,
        "hybrid_alpha": 0.42, "use_reranker": False, "rerank_top_k": 5,
    }
    p = suggest_retrieval_params(optuna.trial.FixedTrial(fixed), ss)
    assert p.chunk_size in ss["chunk_size"]["values"]
    assert p.chunk_overlap in ss["chunk_overlap"]["values"]
    assert ss["top_k"]["low"] <= p.top_k <= ss["top_k"]["high"]
    assert ss["hybrid_alpha"]["low"] <= p.hybrid_alpha <= ss["hybrid_alpha"]["high"]
    assert p.rerank_top_k in ss["rerank_top_k"]["values"]


def test_suggest_unknown_type_raises():
    trial = optuna.trial.FixedTrial({"chunk_size": 256})
    with pytest.raises(ValueError):
        suggest_retrieval_params(trial, {"chunk_size": {"type": "bogus"}})
