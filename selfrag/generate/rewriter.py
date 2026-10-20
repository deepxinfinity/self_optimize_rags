from __future__ import annotations

from typing import Optional

from ..models.base import LLM
from ..models.ollama_adapter import ModelOutputError
from ..utils.timing import TimingLog, maybe_timed


class Rewriter:
    def __init__(self, llm: LLM, rewrite_prompt: str) -> None:
        self.llm = llm
        self.rewrite_prompt = rewrite_prompt.strip()

    def rewrite(self, query: str, timing: Optional[TimingLog] = None) -> str:
        with maybe_timed(timing, "rewrite_query"):
            try:
                out = self.llm.generate(f"Question: {query}", system=self.rewrite_prompt).strip()
            except ModelOutputError:
                out = ""
        return out or query
