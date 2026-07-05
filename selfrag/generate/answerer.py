"""Answer generation: context + question -> answer.

The answer prompt is a *tunable parameter* (search_space.prompts.answer_prompt_seed), optimized in
Stage 2. Here it is just the seed prompt for the baseline.
"""

from __future__ import annotations

from typing import Optional

from ..models.base import LLM
from ..retrieve.pipeline import RetrievalResult


class Answerer:
    def __init__(self, llm: LLM, answer_prompt: str) -> None:
        self.llm = llm
        self.answer_prompt = answer_prompt.strip()

    @staticmethod
    def format_context(results: list[RetrievalResult]) -> str:
        return "\n\n".join(
            f"[{i + 1}] ({r.chunk.chunk_id})\n{r.chunk.text}"
            for i, r in enumerate(results)
        )

    def answer(self, question: str, results: list[RetrievalResult], timing=None) -> str:
        context = self.format_context(results)
        user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        def _gen():
            return self.llm.generate(user, system=self.answer_prompt).strip()

        if timing is not None:
            with timing.timed("generate", n_context_chunks=len(results)):
                return _gen()
        return _gen()
