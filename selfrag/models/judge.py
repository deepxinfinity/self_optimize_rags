from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..utils.cache import DiskCache
from .ollama_adapter import ModelOutputError

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)

# Each rubric only sees the fields it needs. Keeping the retrieved context out of the
# relevancy/correctness prompts makes those two calls several times cheaper.
_RUBRICS: dict[str, dict[str, Any]] = {
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
        "fields": ("question", "context", "answer"),
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
        "fields": ("question", "answer"),
    },
    "answer_correctness": {
        "system": (
            "You are a strict evaluator. Compare the ANSWER with the REFERENCE answer to the "
            "QUESTION. Judge only whether the answer states the same facts as the reference; "
            "ignore wording, length and style."
        ),
        "criteria": (
            "5 = conveys every key fact in the reference; "
            "3 = partially correct, or misses a key fact; "
            "1 = wrong, contradicts the reference, or declines to answer."
        ),
        "fields": ("question", "reference", "answer"),
    },
}

METRICS = tuple(_RUBRICS)


def _normalize(raw_score: float) -> float:
    s = max(1.0, min(5.0, float(raw_score)))
    return round((s - 1.0) / 4.0, 4)


class LLMJudge:
    def __init__(self, llm: Any, model_name: str, cache_dir: str = "./.cache/api") -> None:
        self.llm = llm
        self.model_name = model_name
        self.cache = DiskCache(cache_dir, namespace="judge")

    def score(self, metric: str, question: str, answer: str,
              context: str = "", reference: str = "") -> Optional[float]:
        """A score in [0, 1], or None if the judge's reply was cut off (left out of averages)."""
        if metric not in _RUBRICS:
            raise ValueError(f"unknown judge metric: {metric!r}")
        rubric = _RUBRICS[metric]
        values = {"question": question, "context": context, "reference": reference,
                  "answer": answer}
        if metric == "answer_correctness" and not reference:
            raise ValueError("answer_correctness needs a reference answer")

        system = (
            f"{rubric['system']}\n\nScoring criteria (1-5): {rubric['criteria']}\n"
            'Respond ONLY with JSON: {"score": <integer 1-5>, "reason": "<one sentence>"}'
        )
        body = "\n\n".join(f"{field.upper()}:\n{values[field]}" for field in rubric["fields"])
        prompt = f"{body}\n\nReturn the JSON judgment now."

        payload = {"metric": metric, "model": self.model_name, "system": system, "prompt": prompt}
        key = self.cache.key(payload)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        try:
            raw = self.llm.generate(prompt, system=system, format="json")
        except ModelOutputError:
            return None
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
        num = re.search(r"-?\d+(?:\.\d+)?", raw)
        if num:
            return _normalize(float(num.group(0)))
        # unparseable counts as the worst score so a broken judge shows up in the numbers
        return 0.0


def build_judge_from_spec(spec: Any, cache_dir: str = "./.cache/api") -> LLMJudge:
    from .factory import build_llm
    d = spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)
    llm = build_llm({k: v for k, v in d.items() if k != "id"}, cache_dir)
    return LLMJudge(llm, model_name=d["name"], cache_dir=cache_dir)
