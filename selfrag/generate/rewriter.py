"""Optional query rewriting: reshape the user's question into a better retrieval query.

The rewrite prompt is a tunable parameter (search_space.prompts.rewrite_prompt_seed). Rewriting is
toggled by ``query_rewrite_enabled`` and is off in the naive baseline / Stage-1 search.
"""

from __future__ import annotations

from ..models.base import LLM


class Rewriter:
    def __init__(self, llm: LLM, rewrite_prompt: str) -> None:
        self.llm = llm
        self.rewrite_prompt = rewrite_prompt.strip()

    def rewrite(self, query: str, timing=None) -> str:
        def _do() -> str:
            out = self.llm.generate(f"Question: {query}", system=self.rewrite_prompt).strip()
            return out or query  # never return empty

        if timing is not None:
            with timing.timed("rewrite_query"):
                return _do()
        return _do()
