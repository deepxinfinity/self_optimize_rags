"""Stage 2: optimize the answer prompt against judged metrics (expensive — LLM in the loop).

Retrieval is frozen (Stage-1 winner). We tune the answerer's system prompt with a manual loop:

    run current prompt -> collect the lowest-scoring (failing) examples ->
    ask a strong "proposer" model for an improved prompt -> keep it only if it scores higher.

This is the spec-sanctioned alternative to DSPy and is the primary path here. ``method: dspy``
falls back to this loop with a note (DSPy's local-ollama integration is intentionally left as a
clean extension point rather than shipped unverified). Sample sizes / trial counts are kept small
and the judge + proposer calls are cached, to respect the API budget once models move to an API.

Each trial records its ``time_sec`` and score into a per-trial JSONL log.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from statistics import mean
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import Config
from ..evalgen.generator import QAPair
from ..models.factory import build_judge, build_llm
from ..params import RetrievalParams
from ..rag import build_rag
from ..utils.cache import DiskCache
from ..utils.runs import append_jsonl

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)

# Score/per-sample pair: a scorer returns (aggregate_value, [per-sample dicts]).
Scorer = Callable[[str], "tuple[float, list[dict[str, Any]]]"]
Proposer = Callable[[str, list[dict[str, Any]]], str]


@dataclass
class Stage2Result:
    best_prompt: str
    best_value: float
    seed_value: float
    metric: str
    n_trials: int
    total_seconds: float


def select_failures(per_sample: list[dict[str, Any]], metric: str, k: int = 3) -> list[dict[str, Any]]:
    """The k lowest-scoring examples under ``metric`` — the material shown to the proposer."""
    return sorted(per_sample, key=lambda s: s.get(metric, 0.0))[:k]


def accept(candidate_value: float, best_value: float) -> bool:
    """Keep a proposed prompt only if it strictly improves the objective."""
    return candidate_value > best_value


_PROPOSER_SYSTEM = (
    "You are a prompt engineer improving the SYSTEM PROMPT for a RAG answerer. The answerer is "
    "given a context and a question and must answer using ONLY the context, staying faithful and "
    "directly relevant. You are shown the current prompt and some failing examples with judge "
    "scores (0-1). Write a single improved system prompt that would fix these failures without "
    "over-specializing to them. Respond ONLY with JSON: {\"prompt\": \"<the improved prompt>\"}"
)


class Stage2Optimizer:
    def __init__(self, cfg: Config, pairs: list[QAPair], frozen_params: RetrievalParams,
                 trial_log_path: Optional[str | Path] = None,
                 scorer: Optional[Scorer] = None, proposer: Optional[Proposer] = None) -> None:
        self.cfg = cfg
        self.pairs = pairs
        self.frozen_params = frozen_params
        stage2 = cfg.optimization.get("stage2_prompts", {})
        self.method = stage2.get("method", "manual")
        self.metric = stage2.get("metric", "faithfulness")
        self.n_trials = int(stage2.get("n_trials", 5))
        self.sample_size = int(stage2.get("sample_size", 0)) or None
        self.seed_prompt = cfg.search_space["prompts"]["answer_prompt_seed"].strip()
        self.trial_log_path = Path(trial_log_path) if trial_log_path else None

        # Dependency injection keeps the loop unit-testable without models.
        self._scorer = scorer or self._build_scorer()
        self._propose = proposer or self._build_proposer()

    # ---- real scorer / proposer (skipped entirely when injected in tests) ----
    def _build_scorer(self) -> Scorer:
        rag = build_rag(self.cfg, self.frozen_params)   # retrieval frozen; only the prompt varies
        judge = build_judge(self.cfg.models.judge, self.cfg.runtime.cache_dir)
        sample = self.pairs[:self.sample_size] if self.sample_size else self.pairs

        def scorer(prompt: str) -> tuple[float, list[dict[str, Any]]]:
            rag.answerer.answer_prompt = prompt.strip()
            per: list[dict[str, Any]] = []
            for p in sample:
                results = rag.pipeline.retrieve(p.question, self.frozen_params)
                answer = rag.answerer.answer(p.question, results)
                context = rag.answerer.format_context(results)
                per.append({
                    "question": p.question, "answer": answer, "context": context,
                    "faithfulness": judge.score("faithfulness", p.question, answer, context),
                    "answer_relevancy": judge.score("answer_relevancy", p.question, answer, context),
                })
            value = mean(s[self.metric] for s in per) if per else 0.0
            return round(value, 4), per

        return scorer

    def _build_proposer(self) -> Proposer:
        spec = self.cfg.optimization.get("stage2_prompts", {}).get("proposer")
        # Default proposer = the judge's (capable, different-family) local model.
        proposer_spec = spec or self.cfg.models.judge.model_dump()
        llm = build_llm({k: v for k, v in proposer_spec.items() if k != "id"},
                        self.cfg.runtime.cache_dir)
        cache = DiskCache(self.cfg.runtime.cache_dir, namespace="stage2/proposer")

        def propose(current_prompt: str, failures: list[dict[str, Any]]) -> str:
            fail_text = "\n\n".join(
                f"- Q: {f['question']}\n  A: {f['answer']}\n  "
                f"faithfulness={f.get('faithfulness')}, relevancy={f.get('answer_relevancy')}"
                for f in failures
            )
            user = (f"CURRENT PROMPT:\n{current_prompt}\n\nFAILING EXAMPLES:\n{fail_text}\n\n"
                    "Return the improved prompt as JSON now.")
            payload = {"system": _PROPOSER_SYSTEM, "user": user}
            key = cache.key(payload)
            cached = cache.get(key)
            if cached is not None:
                return cached
            raw = llm.generate(user, system=_PROPOSER_SYSTEM, format="json")
            new_prompt = self._parse_prompt(raw) or current_prompt
            cache.set(key, new_prompt)
            return new_prompt

        return propose

    @staticmethod
    def _parse_prompt(raw: str) -> Optional[str]:
        m = _JSON_OBJ.search(raw)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj.get("prompt"), str) and obj["prompt"].strip():
                    return obj["prompt"].strip()
            except json.JSONDecodeError:
                pass
        return None

    def run(self) -> Stage2Result:
        if self.method == "dspy":
            print("stage2: method 'dspy' not wired — falling back to the manual loop "
                  "(equivalent objective). Set method: manual to silence this.", file=sys.stderr)

        t0 = time.perf_counter()
        best_prompt = self.seed_prompt
        best_value, best_per = self._scorer(best_prompt)
        seed_value = best_value
        self._log(0, best_prompt, best_value, round(time.perf_counter() - t0, 4))

        for i in range(1, self.n_trials + 1):
            ti = time.perf_counter()
            failures = select_failures(best_per, self.metric)
            candidate = self._propose(best_prompt, failures)
            cand_value, cand_per = self._scorer(candidate)
            if accept(cand_value, best_value):
                best_prompt, best_value, best_per = candidate, cand_value, cand_per
            self._log(i, candidate, cand_value, round(time.perf_counter() - ti, 4))

        return Stage2Result(
            best_prompt=best_prompt,
            best_value=round(best_value, 4),
            seed_value=round(seed_value, 4),
            metric=self.metric,
            n_trials=self.n_trials,
            total_seconds=round(time.perf_counter() - t0, 4),
        )

    def _log(self, trial: int, prompt: str, value: float, elapsed: float) -> None:
        if self.trial_log_path is not None:
            append_jsonl(self.trial_log_path, {
                "trial": trial, "metric": self.metric, "value": value,
                "time_sec": elapsed, "prompt": prompt,
            })
