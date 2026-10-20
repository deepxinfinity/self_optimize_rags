from __future__ import annotations

import numpy as np
import pytest

from selfrag.config import _parse_scalar, load_config
from selfrag.corpus.chunker import _chunk_text, chunk_documents
from selfrag.corpus.loader import Document, _extensions_from_glob
from selfrag.index.bm25 import BM25Index
from selfrag.index.hybrid import _minmax
from selfrag.index.vector_store import VectorStore
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
        load_config(CONFIG, ["splits.test=0.5"])


def test_both_configs_load():
    for path in ("configs/default.yaml", "configs/qasper.yaml"):
        cfg = load_config(path)
        assert cfg.models.generator.model_dump().get("num_ctx")
        baseline_retrieval_params(cfg)


def test_glob_brace_expansion():
    assert set(_extensions_from_glob("**/*.{txt,md,pdf}")) == {".txt", ".md", ".pdf"}
    assert _extensions_from_glob("**/*.txt") == [".txt"]


def test_chunk_overlap():
    chunks = _chunk_text("one two three four five six seven eight", chunk_size=4, overlap=1)
    assert chunks[0].split()[-1] == chunks[1].split()[0]


def test_chunk_without_overlap():
    assert _chunk_text("a b c d e", chunk_size=2, overlap=0) == ["a b", "c d", "e"]


def test_chunk_rejects_overlap_ge_size():
    with pytest.raises(ValueError):
        _chunk_text("a b c d", chunk_size=3, overlap=3)


def test_chunk_documents_ids():
    docs = [Document("d.md", "d.md", "alpha beta gamma delta epsilon zeta")]
    chunks = chunk_documents(docs, chunk_size=3, chunk_overlap=1)
    assert chunks[0].chunk_id == "d.md::0"
    assert all(c.doc_id == "d.md" for c in chunks)


def test_minmax_extremes():
    assert np.allclose(_minmax(np.array([1.0, 1.0, 1.0])), 0.0)
    assert np.allclose(_minmax(np.array([0.0, 5.0, 10.0])), [0.0, 0.5, 1.0])


def test_bm25_ranks_keyword_match():
    bm = BM25Index(["dense vector embeddings cosine", "bm25 keyword inverted index",
                    "positional encoding"])
    assert bm.query("keyword index", k=1)[0][0] == 1


def test_vector_store_orders_by_cosine():
    vecs = np.eye(4, dtype=np.float32)
    q = np.array([0.1, 0.9, 0.3, 0.0], dtype=np.float32)
    vs = VectorStore(vecs)
    assert [i for i, _ in vs.query(q, 3)] == [1, 2, 0]
    assert vs.query(q, 10)[0][1] == pytest.approx(0.9)
    assert np.allclose(vs.scores(q), q)


def test_baseline_section_in_both_configs_matches():
    a = baseline_retrieval_params(load_config("configs/default.yaml"))
    b = baseline_retrieval_params(load_config("configs/qasper.yaml"))
    assert a == b and (a.chunk_size, a.top_k, a.use_reranker) == (256, 5, False)


def test_derived_baseline_when_config_has_none():
    cfg = load_config(CONFIG)
    cfg.baseline = {}
    p = baseline_retrieval_params(cfg)
    assert p.use_reranker is False and p.query_rewrite_enabled is False
    assert p.chunk_size == cfg.search_space["chunk_size"]["values"][0]
    assert p.top_k == cfg.search_space["top_k"]["low"]
