from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pytest

from selfrag.config import load_config
from selfrag.params import RetrievalParams
from selfrag.rag import RagFactory
from selfrag.utils.timing import TimingLog

QASPER_CONFIG = "configs/qasper.yaml"


class CountingEmbedder:
    """Deterministic bag-of-hashed-words vectors, so dense search behaves sensibly in tests."""

    name = "counting"

    def __init__(self, dim: int = 256) -> None:
        self.calls = 0
        self.texts_embedded = 0
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.texts_embedded += len(texts)
        vecs = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                vecs[i, zlib.crc32(w.encode()) % self._dim] += 1.0
        vecs += 1e-6
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    @property
    def dim(self) -> int:
        return self._dim


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    d = tmp_path / "corpus"
    d.mkdir()
    for i in range(3):
        (d / f"doc{i}.txt").write_text(" ".join(f"word{i}x{j}" for j in range(400)))
    return d


class LengthReranker:
    name = "length"

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        return [float(len(p)) for p in passages]


def make_factory(corpus: Path, tmp_path: Path, *overrides: str) -> RagFactory:
    cfg = load_config(QASPER_CONFIG, [f"dataset.corpus_path={corpus}",
                                      f"runtime.embedding_cache_dir={tmp_path / 'emb'}",
                                      *overrides])
    f = RagFactory(cfg, TimingLog())
    f._embedder = CountingEmbedder()
    f._reranker = LengthReranker()
    return f


@pytest.fixture
def factory(corpus: Path, tmp_path: Path) -> RagFactory:
    return make_factory(corpus, tmp_path)


def params(chunk_size: int = 128, overlap: int = 32, **kw) -> RetrievalParams:
    base = dict(chunk_size=chunk_size, chunk_overlap=overlap, top_k=3, hybrid_alpha=0.5,
                use_reranker=False, rerank_candidates=20, query_rewrite_enabled=False)
    base.update(kw)
    return RetrievalParams(**base)
