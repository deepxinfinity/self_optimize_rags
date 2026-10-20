"""Stage 2: tune the answer prompt with retrieval frozen.

Manual loop: show a proposer model the current prompt and some failing train questions, score
its rewrite on val, and keep it only if it beats the current best by more than `min_delta`.
Failures come from train and acceptance from val, so the proposer never sees the questions
that decide whether its prompt is kept.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import Config
from ..evalgen.generator import QAPair
from ..evaluate import evaluate_answers
from ..models.factory import build_judge, build_llm
from ..models.ollama_adapter import ModelOutputError
from ..params import RetrievalParams
from ..rag import RagFactory
from ..utils.cache import DiskCache
from ..utils.runs import append_jsonl

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)

Scorer = Callable[[str, list[QAPair]], "tuple[float, list[dict[str, Any]]]"]
Proposer = Callable[[str, list[dict[str, Any]]], str]


@dataclass
class Stage2Result:
    best_prompt: str
    best_value: float
    seed_value: float
    metric: str
    n_trials: int
    total_seconds: float


def select_failures(per_sample: list[dict[str, Any]], metric: str, k: int = 3,
                    offset: int = 0) -> list[dict[str, Any]]:
    failing = sorted((s for s in per_sample if (s.get(metric) or 0.0) < 1.0),
                     key=lambda s: s.get(metric) or 0.0)
    if not failing:
        return []
    start = offset % len(failing)
    return (failing[start:] + failing[:start])[:k]


def accept(candidate_value: float, best_value: float, min_delta: float = 0.0) -> bool:
    return candidate_value - best_value > min_delta


_PROPOSER_SYSTEM = (
    "You are a prompt engineer improving the SYSTEM PROMPT for a RAG answerer. The answerer is "
    "given a context and a question and must answer using ONLY the context. You are shown the "
    "current prompt and some questions it handled badly, with the reference answer and a judge "
    "score (0-1). Write a single improved system prompt that would fix these failures without "
    "over-specializing to them. Respond ONLY with JSON: {\"prompt\": \"<the improved prompt>\"}"
)


class Stage2Optimizer:
    def __init__(self, cfg: Config, val_pairs: list[QAPair], frozen_params: RetrievalParams,
                 train_pairs: Optional[list[QAPair]] = None,
                 trial_log_path: Optional[str | Path] = None,
                 factory: Optional[RagFactory] = None, judge: Any = None,
                 scorer: Optional[Scorer] = None, proposer: Optional[Proposer] = None,
                 log: Optional[Callable[[str], None]] = None) -> None:
        self.cfg = cfg
        self.frozen_params = frozen_params
        stage2 = cfg.optimization.get("stage2_prompts", {})
        self.method = stage2.get("method", "manual")
        self.metric = stage2.get("metric", "answer_correctness")
        self.n_trials = int(stage2.get("n_trials", 5))
        self.min_delta = float(stage2.get("min_delta", 0.0))
        self.n_failures = int(stage2.get("n_failures", 3))
        val_n = int(stage2.get("sample_size") or 0) or None
        train_n = int(stage2.get("train_sample_size") or 0) or None
        self.val = val_pairs[:val_n]
        self.train = (train_pairs or [])[:train_n]
        self.seed_prompt = cfg.search_space["prompts"]["answer_prompt_seed"].strip()
        self.trial_log_path = Path(trial_log_path) if trial_log_path else None
        self.log = log or (lambda msg: None)

        self._scorer = scorer or self._build_scorer(factory, judge)
        self._propose = proposer or self._build_proposer()

    def _build_scorer(self, factory: Optional[RagFactory], judge: Any) -> Scorer:
        factory = factory or RagFactory(self.cfg)
        judge = judge or build_judge(self.cfg.models.judge, self.cfg.runtime.cache_dir)

        def scorer(prompt: str, pairs: list[QAPair]) -> tuple[float, list[dict[str, Any]]]:
            rag = factory.build(self.frozen_params, prompt)
            summary, items = evaluate_answers(rag, judge, pairs, metrics=(self.metric,),
                                              timing=factory.timing)
            for item, p in zip(items, pairs):
                item["reference"] = p.answer
            return summary.get(self.metric, 0.0), items

        return scorer

    def _build_proposer(self) -> Proposer:
        spec = self.cfg.optimization.get("stage2_prompts", {}).get("proposer")
        proposer_spec = spec or self.cfg.models.judge.model_dump()
        llm = build_llm({k: v for k, v in proposer_spec.items() if k != "id"},
                        self.cfg.runtime.cache_dir)
        cache = DiskCache(self.cfg.runtime.cache_dir, namespace="stage2/proposer")

        def propose(current_prompt: str, failures: list[dict[str, Any]]) -> str:
            fail_text = "\n\n".join(
                f"- Q: {f['question']}\n  Reference: {f.get('reference', '')}\n"
                f"  Answer given: {f['answer']}\n  {self.metric}={f.get(self.metric)}"
                for f in failures
            )
            user = (f"CURRENT PROMPT:\n{current_prompt}\n\nFAILING EXAMPLES:\n{fail_text}\n\n"
                    "Return the improved prompt as JSON now.")
            key = cache.key({"system": _PROPOSER_SYSTEM, "user": user})
            cached = cache.get(key)
            if cached is not None:
                return cached
            try:
                raw = llm.generate(user, system=_PROPOSER_SYSTEM, format="json")
            except ModelOutputError:
                return current_prompt
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

    def _failure_pool(self, prompt: str) -> list[dict[str, Any]]:
        _, items = self._scorer(prompt, self.train or self.val)
        return items

    def run(self) -> Stage2Result:
        if self.method != "manual":
            print(f"stage2: method {self.method!r} is not implemented, using the manual loop",
                  file=sys.stderr)
        if not self.train:
            print("stage2: no train pairs given, failures will be mined from val", file=sys.stderr)

        t0 = time.perf_counter()
        best_prompt = self.seed_prompt
        best_value, _ = self._scorer(best_prompt, self.val)
        seed_value = best_value
        self._log(0, best_prompt, best_value, True, t0)
        self.log(f"  seed prompt: {self.metric}={best_value:.3f} on {len(self.val)} val pairs")

        pool = self._failure_pool(best_prompt) if self.n_trials else []
        tried = {best_prompt}
        offset = 0
        n_run = 0
        for i in range(1, self.n_trials + 1):
            ti = time.perf_counter()
            failures = select_failures(pool, self.metric, self.n_failures, offset)
            if not failures:
                break
            n_run = i
            # a rejected candidate would come back unchanged from the same failures, so each
            # trial shows the proposer a different slice of them
            offset += self.n_failures
            candidate = self._propose(best_prompt, failures).strip()
            if candidate in tried:
                self._log(i, candidate, None, False, ti, note="duplicate")
                self.log(f"  trial {i}/{self.n_trials}: proposer repeated an earlier prompt, "
                         "skipped")
                continue
            tried.add(candidate)
            value, _ = self._scorer(candidate, self.val)
            accepted = accept(value, best_value, self.min_delta)
            self._log(i, candidate, value, accepted, ti)
            self.log(f"  trial {i}/{self.n_trials}: {self.metric}={value:.3f}"
                     f" vs best {best_value:.3f} -> {'accepted' if accepted else 'rejected'}"
                     f"  ({time.perf_counter() - ti:.0f}s)")
            if accepted:
                best_prompt, best_value = candidate, value
                pool, offset = self._failure_pool(best_prompt), 0

        return Stage2Result(
            best_prompt=best_prompt,
            best_value=round(best_value, 4),
            seed_value=round(seed_value, 4),
            metric=self.metric,
            n_trials=n_run,
            total_seconds=round(time.perf_counter() - t0, 4),
        )

    def _log(self, trial: int, prompt: str, value: Optional[float], accepted: bool,
             started: float, note: str = "") -> None:
        if self.trial_log_path is None:
            return
        record: dict[str, Any] = {
            "trial": trial, "metric": self.metric, "value": value, "accepted": accepted,
            "time_sec": round(time.perf_counter() - started, 4), "prompt": prompt,
        }
        if note:
            record["note"] = note
        append_jsonl(self.trial_log_path, record)
