"""Provider-agnostic model interfaces.

A thin adapter layer so ollama, sentence-transformers, and an HTTP API are interchangeable behind
one interface. Arms and the judge are just model configs routed through the same factory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class LLM(ABC):
    """Text-in, text-out generator (RAG answerer, query rewriter, eval-set generator arms)."""

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str: ...


class Embedder(ABC):
    """Dense embeddings for retrieval."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalized embeddings."""

    @property
    @abstractmethod
    def dim(self) -> int: ...


class Reranker(ABC):
    """Cross-encoder that rescores (query, passage) pairs."""

    @abstractmethod
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Return a relevance score per passage, aligned with the input order."""


class Judge(ABC):
    """LLM-as-judge for faithfulness / answer relevancy. Implemented in Increment 3."""

    @abstractmethod
    def score(self, metric: str, question: str, answer: str, context: str) -> float: ...
