from __future__ import annotations

from typing import Optional

from ..models.base import LLM
from ..models.ollama_adapter import ModelOutputError
from ..retrieve.pipeline import RetrievalResult
from ..utils.timing import TimingLog, maybe_timed


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

    def answer(self, question: str, results: list[RetrievalResult],
               timing: Optional[TimingLog] = None) -> str:
        user = f"Context:\n{self.format_context(results)}\n\nQuestion: {question}\n\nAnswer:"
        with maybe_timed(timing, "generate", n_context_chunks=len(results)):
            try:
                return self.llm.generate(user, system=self.answer_prompt).strip()
            except ModelOutputError:
                # a reply that looped until ollama stopped it counts as no answer
                return ""
