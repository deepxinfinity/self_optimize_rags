"""sentence-transformers adapters: dense embeddings + cross-encoder reranker.

Models are loaded lazily on first use so that building a pipeline that never reranks doesn't pay
the cost of downloading/loading the cross-encoder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _hf_cache_dir() -> Path:
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    return Path.home() / ".cache" / "huggingface" / "hub"


def _prefer_cache(model_name: str) -> None:
    """If a model is already in the HF cache, load it offline — skips the network HEAD
    update-checks that otherwise stall (and get 429-throttled) on every load. A model that
    isn't cached still downloads normally. Explicit HF_HUB_OFFLINE is always respected."""
    if os.environ.get("HF_HUB_OFFLINE"):
        return
    cached = _hf_cache_dir() / f"models--{model_name.replace('/', '--')}"
    if cached.is_dir():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class STEmbedder:
    def __init__(self, name: str, **_: Any) -> None:
        self.name = name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            _prefer_cache(self.name)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,   # cosine similarity == inner product on unit vectors
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vecs, dtype=np.float32)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


class STReranker:
    def __init__(self, name: str, **_: Any) -> None:
        self.name = name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            _prefer_cache(self.name)
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.name)
        return self._model

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]
