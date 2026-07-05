"""LLM-as-judge for faithfulness / answer_relevancy.

The judge is intentionally a *different model family* from the answerer to avoid self-evaluation
bias (config default: llama3.1:8b judging a qwen answerer). Judge quality directly determines how
trustworthy every judged score is — this is the #1 role to upgrade to a strong API judge later.

Every judge call is cached to disk keyed by its full input, regardless of provider, so re-runs are
free and reproducible (protects the API budget once the judge moves to an API model).
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..utils.cache import DiskCache

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)

# Metric rubrics. Each asks for an integer 1-5, normalized to [0,1] as (score-1)/4.
_RUBRICS: dict[str, dict[str, str]] = {
    "faithfulness": {
        "system": (
            "You are a strict evaluator. Judge whether the ANSWER is faithful to the CONTEXT: "
            "every claim in the answer must be supported by the context, with no fabrication or "
            "outside knowledge. The question is provided only for reference."
        ),
        "criteria": (
            "5 = every claim fully supported by the context; "
            "3 = mostly supported but with a minor unsupported detail; "
            "1 = contains claims contradicted by or absent from the context."
        ),
    },
    "answer_relevancy": {
        "system": (
            "You are a strict evaluator. Judge whether the ANSWER actually addresses the QUESTION: "
            "is it on-topic, direct, and complete? Ignore whether it is factually correct here."
        ),
        "criteria": (
            "5 = directly and completely answers the question; "
            "3 = partially answers or is padded with irrelevant content; "
            "1 = does not address the question."
        ),
    },
}


def _normalize(raw_score: float) -> float:
    """Map an integer 1-5 rubric score to [0, 1]. Clamps out-of-range values.

    The rubric always asks for 1-5, so we treat the score on that scale unconditionally (a bare
    "1" means the worst rating, not a 0-1 probability).
    """
    s = max(1.0, min(5.0, float(raw_score)))
    return round((s - 1.0) / 4.0, 4)


class LLMJudge:
    def __init__(self, llm: Any, model_name: str, cache_dir: str = "./.cache/api") -> None:
        self.llm = llm
        self.model_name = model_name
        self.cache = DiskCache(cache_dir, namespace="judge")

    def score(self, metric: str, question: str, answer: str, context: str) -> float:
        if metric not in _RUBRICS:
            raise ValueError(f"unknown judge metric: {metric!r}")
        rubric = _RUBRICS[metric]
        system = (
            f"{rubric['system']}\n\nScoring criteria (1-5): {rubric['criteria']}\n"
            'Respond ONLY with JSON: {"score": <integer 1-5>, "reason": "<one sentence>"}'
        )
        prompt = (
            f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
            "Return the JSON judgment now."
        )
        payload = {
            "metric": metric, "model": self.model_name, "system": system, "prompt": prompt,
        }
        key = self.cache.key(payload)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        raw = self.llm.generate(prompt, system=system, format="json")
        score = self._parse(raw)
        self.cache.set(key, score, meta={"metric": metric})
        return score

    def _parse(self, raw: str) -> float:
        m = _JSON_OBJ.search(raw)
        if m:
            try:
                obj = json.loads(m.group(0))
                if "score" in obj:
                    return _normalize(float(obj["score"]))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        # Fallback: first bare number in the text.
        num = re.search(r"-?\d+(?:\.\d+)?", raw)
        if num:
            return _normalize(float(num.group(0)))
        return 0.0  # unparseable -> worst score, surfaces judge issues rather than hiding them


def build_judge_from_spec(spec: Any, cache_dir: str = "./.cache/api") -> LLMJudge:
    from .factory import build_llm  # local import to avoid a circular import at module load
    d = spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)
    llm = build_llm({k: v for k, v in d.items() if k != "id"}, cache_dir)
    return LLMJudge(llm, model_name=d["name"], cache_dir=cache_dir)
