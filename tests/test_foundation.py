"""Increment-1 foundation tests. Fast: no model downloads, no ollama."""

from __future__ import annotations

import numpy as np
import pytest

from selfrag.config import load_config, _parse_scalar
from selfrag.corpus.chunker import _chunk_text, chunk_documents
from selfrag.corpus.loader import Document, _extensions_from_glob
from selfrag.index.bm25 import BM25Index
from selfrag.index.hybrid import _minmax
from selfrag.params import baseline_retrieval_params

CONFIG = "configs/default.yaml"


def test_set_override_typing():
    cfg = load_config(CONFIG, ["search_space.top_k.low=6", "models.generator.name=llama3.2:3b"])
    assert cfg.search_space["top_k"]["low"] == 6
    assert isinstance(cfg.search_space["top_k"]["low"], int)
    assert cfg.models.generator.name == "llama3.2:3b"


@pytest.mark.parametrize("raw,expected", [
    ("5", 5), ("0.7", 0.7), ("true", True), ("false", False),
    ("llama3.2:3b", "llama3.2:3b"), ("./data/x", "./data/x"),
])
def test_parse_scalar(raw, expected):
    assert _parse_scalar(raw) == expected


def test_splits_must_sum_to_one():
    with pytest.raises(Exception):
        load_config(CONFIG, ["splits.test=0.5"])  # 0.6+0.2+0.5 != 1.0


def test_glob_brace_expansion():
    assert set(_extensions_from_glob("**/*.{txt,md,pdf}")) == {".txt", ".md", ".pdf"}
    assert _extensions_from_glob("**/*.txt") == [".txt"]


def test_chunk_overlap():
    chunks = _chunk_text("one two three four five six seven eight", chunk_size=4, overlap=1)
    assert chunks[0].split()[-1] == chunks[1].split()[0]  # 1-token overlap


def test_chunk_rejects_overlap_ge_size():
    with pytest.raises(ValueError):
        _chunk_text("a b c d", chunk_size=3, overlap=3)


def test_chunk_documents_ids():
    docs = [Document("d.md", "d.md", "alpha beta gamma delta epsilon zeta")]
    chunks = chunk_documents(docs, chunk_size=3, chunk_overlap=1)
    assert chunks[0].chunk_id == "d.md::0"
    assert all(c.doc_id == "d.md" for c in chunks)


def test_minmax_extremes():
    assert np.allclose(_minmax(np.array([1.0, 1.0, 1.0])), 0.0)  # constant -> zeros, no div0
    assert np.allclose(_minmax(np.array([0.0, 5.0, 10.0])), [0.0, 0.5, 1.0])


def test_bm25_ranks_keyword_match():
    texts = ["dense vector embeddings cosine", "bm25 keyword inverted index", "positional encoding"]
    bm = BM25Index(texts)
    hits = bm.query("keyword index", k=1)
    assert hits[0][0] == 1


def test_baseline_params_naive_toggles_off():
    cfg = load_config(CONFIG)
    p = baseline_retrieval_params(cfg)
    assert p.use_reranker is False and p.query_rewrite_enabled is False
    assert p.chunk_size in cfg.search_space["chunk_size"]["values"]
    assert p.top_k == cfg.search_space["top_k"]["low"]
