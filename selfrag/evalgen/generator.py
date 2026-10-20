"""Synthetic QA generation for one eval-generator arm.

Generation is kept in our hands (passage sampling, prompt, caching) because the generator is
the variable the study compares. Each pair records the gold document plus a verbatim evidence
quote; pairs whose quote cannot be found in the source passage are dropped.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from ..config import Config
from ..corpus.chunker import Chunk, chunk_documents
from ..corpus.loader import document_title, load_corpus
from ..corpus.qasper import scope_question
from ..metrics.retrieval import ngrams
from ..models.factory import build_llm
from ..models.ollama_adapter import ModelOutputError
from ..utils.cache import DiskCache
from ..utils.timing import TimingLog, maybe_timed

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM = (
    "You write one evaluation question-answer pair from a passage of a document. "
    "The question must be answerable from the passage alone, specific, and self-contained "
    "(never refer to 'the passage' or 'the text'). The answer must be short and fully supported "
    "by the passage. Also copy, word for word, the sentence from the passage that supports "
    'the answer. Respond with JSON only: {"question": "...", "answer": "...", "evidence": "..."}'
)

GROUNDING_THRESHOLD = 0.8


@dataclass
class QAPair:
    id: str
    arm_id: str
    question: str
    answer: str
    gold_doc_id: str
    gold_chunk_id: str = ""
    source_text: str = ""
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_grounded(evidence: str, passage: str, threshold: float = GROUNDING_THRESHOLD) -> bool:
    ev = ngrams(evidence)
    return bool(ev) and len(ev & ngrams(passage)) / len(ev) >= threshold


def build_eval_chunks(cfg: Config) -> list[Chunk]:
    gen_size = int(cfg.evalgen.get("gen_chunk_size", 512))
    gen_overlap = int(cfg.evalgen.get("gen_chunk_overlap", 64))
    docs = load_corpus(cfg.dataset.corpus_path, cfg.dataset.corpus_glob)
    return chunk_documents(docs, chunk_size=gen_size, chunk_overlap=gen_overlap)


class EvalGenerator:
    def __init__(self, arm_spec: Any, cache_dir: str = "./.cache/api", seed: int = 42) -> None:
        d = arm_spec.model_dump() if hasattr(arm_spec, "model_dump") else dict(arm_spec)
        self.arm_id = d["id"]
        self.temperature = d.get("temperature", 0.7)
        self.llm = build_llm({k: v for k, v in d.items() if k != "id"}, cache_dir)
        self.cache = DiskCache(cache_dir, namespace=f"evalgen/{self.arm_id}")
        self.seed = seed
        self.stats: Counter = Counter()

    @staticmethod
    def _parse(raw: str) -> Optional[dict[str, str]]:
        m = _JSON_OBJ.search(raw)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        fields_ = {k: obj.get(k) for k in ("question", "answer", "evidence")}
        if not all(isinstance(v, str) and v.strip() for v in fields_.values()):
            return None
        return {k: v.strip() for k, v in fields_.items()}

    def _generate_one(self, chunk: Chunk, variant: int) -> dict[str, str]:
        """The parsed pair, {} if the reply didn't parse, or {"error": "aborted"}."""
        # the variant nonce lets one passage yield several different questions
        payload = {
            "arm": self.arm_id,
            "model": self.llm.name,
            "system": _SYSTEM,
            "passage": chunk.text,
            "temperature": self.temperature,
            "variant": variant,
        }
        key = self.cache.key(payload)
        hit = self.cache.get(key)
        if hit is not None:
            return hit

        prompt = f"Passage:\n{chunk.text}\n\nGenerate the JSON question-answer pair now."
        try:
            raw = self.llm.generate(prompt, system=_SYSTEM, format="json",
                                    seed=self.seed + variant)
            result = self._parse(raw) or {}
        except ModelOutputError:
            result = {"error": "aborted"}
        # failures are cached too, so a re-run doesn't ask the model again
        self.cache.set(key, result, meta={"chunk_id": chunk.chunk_id})
        return result

    def generate(self, chunks: list[Chunk], n_pairs: int, timing: Optional[TimingLog] = None,
                 log: Optional[Callable[[str], None]] = None,
                 log_every: int = 20) -> list[QAPair]:
        if not chunks:
            raise ValueError("no chunks to generate from; is the corpus empty?")
        order = list(range(len(chunks)))
        random.Random(self.seed).shuffle(order)

        pairs: list[QAPair] = []
        seen: set[str] = set()
        attempt = 0
        started = time.perf_counter()
        while len(pairs) < n_pairs and attempt < n_pairs * 3:
            if log and attempt and (attempt == 1 or attempt % log_every == 0):
                log(self._progress(len(pairs), n_pairs, attempt, time.perf_counter() - started))
            chunk = chunks[order[attempt % len(order)]]
            variant = attempt // len(order)
            attempt += 1
            with maybe_timed(timing, "evalgen_call", arm=self.arm_id):
                parsed = self._generate_one(chunk, variant)

            if parsed.get("error") == "aborted":
                self.stats["aborted"] += 1
                continue
            if not parsed:
                self.stats["unparsed"] += 1
                continue
            if not is_grounded(parsed["evidence"], chunk.text):
                self.stats["ungrounded"] += 1
                continue
            norm = " ".join(parsed["question"].lower().split())
            if norm in seen:
                self.stats["duplicate"] += 1
                continue
            seen.add(norm)
            pairs.append(QAPair(
                id=f"{self.arm_id}-{len(pairs):05d}",
                arm_id=self.arm_id,
                question=parsed["question"],
                answer=parsed["answer"],
                gold_doc_id=chunk.doc_id,
                gold_chunk_id=chunk.chunk_id,
                source_text=chunk.text,
                evidence=[parsed["evidence"]],
            ))
        self.stats["attempts"] = attempt
        self.stats["kept"] = len(pairs)
        return pairs

    def _progress(self, kept: int, target: int, attempts: int, elapsed: float) -> str:
        per_call = elapsed / attempts
        line = (f"  {self.arm_id}: {kept}/{target} pairs after {attempts} calls "
                f"({self.stats['unparsed']} unparsed, {self.stats['aborted']} aborted, "
                f"{self.stats['ungrounded']} ungrounded, {self.stats['duplicate']} duplicate), "
                f"{per_call:.1f}s/call")
        if kept:
            line += f", ~{(target - kept) * elapsed / kept / 60:.0f} min left"
        return line


def generate_arm_evalset(cfg: Config, arm_spec: Any, chunks: list[Chunk],
                         timing: Optional[TimingLog] = None,
                         log: Optional[Callable[[str], None]] = None,
                         ) -> tuple[list[QAPair], dict[str, int]]:
    n_pairs = int(cfg.evalgen.get("n_pairs_per_arm", 100))
    gen = EvalGenerator(arm_spec, cache_dir=cfg.runtime.cache_dir, seed=cfg.splits.seed)
    pairs = gen.generate(chunks, n_pairs=n_pairs, timing=timing, log=log)
    if cfg.evalgen.get("scope_with_title"):
        # same wording as the scoped real questions, so both sides look alike to retrieval
        docs = load_corpus(cfg.dataset.corpus_path, cfg.dataset.corpus_glob)
        titles = {d.doc_id: document_title(d) for d in docs}
        for p in pairs:
            p.question = scope_question(p.question, titles[p.gold_doc_id])
    return pairs, dict(gen.stats)
