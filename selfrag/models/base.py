from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str: ...


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """(n, d) float32, L2-normalised."""

    @property
    @abstractmethod
    def dim(self) -> int: ...


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """One relevance score per passage, in input order."""


class Judge(ABC):
    @abstractmethod
    def score(self, metric: str, question: str, answer: str,
              context: str = "", reference: str = "") -> float: ...
