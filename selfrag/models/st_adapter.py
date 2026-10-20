from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _hf_cache_dir() -> Path:
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    return Path.home() / ".cache" / "huggingface" / "hub"


def _prefer_cache(model_name: str) -> None:
    # Load cached models offline. Otherwise every load does HEAD requests to the hub,
    # which stall and get rate limited during long runs.
    if os.environ.get("HF_HUB_OFFLINE"):
        return
    cached = _hf_cache_dir() / f"models--{model_name.replace('/', '--')}"
    if cached.is_dir():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class STEmbedder:
    slice_size = 1024

    def __init__(self, name: str, device: Optional[str] = None, **_: Any) -> None:
        self.name = name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            _prefer_cache(self.name)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name, device=self.device)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        parts = []
        for i in range(0, max(len(texts), 1), self.slice_size):
            parts.append(self.model.encode(
                texts[i:i + self.slice_size],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ))
            if len(texts) > 1 and self.model.device.type == "mps":
                import torch
                # otherwise the MPS allocator keeps every batch's buffers and the process
                # grows to several GB
                torch.mps.empty_cache()
        return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


class STReranker:
    def __init__(self, name: str, device: Optional[str] = None, **_: Any) -> None:
        self.name = name
        self.device = device
        self._model = None
        # Stage 1 trials on the same chunking rerank mostly the same (query, chunk) pairs
        self._scores: dict[tuple[str, str], float] = {}

    @property
    def model(self):
        if self._model is None:
            _prefer_cache(self.name)
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.name, device=self.device)
        return self._model

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        keys = [(query, hashlib.sha1(p.encode()).hexdigest()) for p in passages]
        todo = {k: p for k, p in zip(keys, passages) if k not in self._scores}
        if todo:
            scores = self.model.predict([[query, p] for p in todo.values()],
                                        show_progress_bar=False)
            self._scores.update(zip(todo, (float(x) for x in scores)))
        return [self._scores[k] for k in keys]
