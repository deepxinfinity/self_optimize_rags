"""Eval-set generation — the experimental variable of the study.

Each *arm* (a swappable model config) generates synthetic QA pairs from the corpus. Generation is
kept fully under our control (not a library default) so the two-arm comparison is fair:
  - We choose the source passages (seeded sampling).
  - We control the prompt.
  - Every call is cached to disk, so re-runs are free and reproducible.

Gold truth is recorded at the *document* level (``gold_doc_id``) plus the source passage text.
This is robust to the Stage-1 ``chunk_size`` sweep (chunk ids change per trial; doc ids don't) and
matches the real_questions.jsonl schema, so synthetic and real eval score the same way.

Generation-time chunking uses a FIXED window (evalgen.gen_chunk_size / gen_chunk_overlap, with
sensible defaults) independent of the search space, so the pool of source passages is stable.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

from ..config import Config
from ..corpus.chunker import Chunk, chunk_documents
from ..corpus.loader import load_corpus
from ..models.factory import build_llm
from ..utils.cache import DiskCache
from ..utils.timing import TimingLog

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM = (
    "You generate a single evaluation question-answer pair from a passage. "
    "The question MUST be answerable using ONLY the passage, and must be specific and "
    "self-contained (do not say 'the passage' or 'according to the text'). "
    "The answer must be concise and fully grounded in the passage. "
    'Respond with a JSON object of exactly this shape: {"question": "...", "answer": "..."}'
)


@dataclass
class QAPair:
    id: str
    arm_id: str
    question: str
    answer: str
    gold_doc_id: str        # retrieval ground truth (robust to chunk_size sweeps)
    gold_chunk_id: str      # generation-time chunk id (informational)
    source_text: str        # the passage the pair was generated from

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_eval_chunks(cfg: Config) -> list[Chunk]:
    """Chunk the corpus at the fixed generation window used for eval-set creation."""
    gen_size = int(cfg.evalgen.get("gen_chunk_size", 512))
    gen_overlap = int(cfg.evalgen.get("gen_chunk_overlap", 64))
    docs = load_corpus(cfg.dataset.corpus_path, cfg.dataset.corpus_glob)
    return chunk_documents(docs, chunk_size=gen_size, chunk_overlap=gen_overlap)


class EvalGenerator:
    def __init__(self, arm_spec: Any, cache_dir: str = "./.cache/api", seed: int = 42) -> None:
        # arm_spec is a pydantic EvalArm (has id, provider, name, temperature, ...)
        d = arm_spec.model_dump() if hasattr(arm_spec, "model_dump") else dict(arm_spec)
        self.arm_id = d["id"]
        self.temperature = d.get("temperature", 0.7)
        # Route the arm through the same model factory as everything else.
        self.llm = build_llm({k: v for k, v in d.items() if k != "id"}, cache_dir)
        self.cache = DiskCache(cache_dir, namespace=f"evalgen/{self.arm_id}")
        self.seed = seed

    def _parse(self, raw: str) -> Optional[dict[str, str]]:
        m = _JSON_OBJ.search(raw)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        q, a = obj.get("question"), obj.get("answer")
        if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
            return {"question": q.strip(), "answer": a.strip()}
        return None

    def _generate_one(self, chunk: Chunk, variant: int) -> Optional[dict[str, str]]:
        # Cache key includes a variant nonce so multiple questions per passage differ but re-runs
        # remain reproducible.
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
        # `seed` makes each variant deterministic; format="json" constrains the output shape.
        raw = self.llm.generate(prompt, system=_SYSTEM, format="json",
                                 seed=self.seed + variant)
        parsed = self._parse(raw)
        self.cache.set(key, parsed, meta={"chunk_id": chunk.chunk_id})
        return parsed

    def generate(self, chunks: list[Chunk], n_pairs: int,
                 timing: Optional[TimingLog] = None) -> list[QAPair]:
        """Produce up to ``n_pairs`` QA pairs, sampling passages round-robin (seeded).

        If there are fewer chunks than requested pairs, passages are revisited with a new variant
        nonce so we still get distinct questions.
        """
        if not chunks:
            raise ValueError("no chunks to generate from — is the corpus empty?")
        rng = random.Random(self.seed)
        order = list(range(len(chunks)))
        rng.shuffle(order)

        pairs: list[QAPair] = []
        attempt = 0
        max_attempts = n_pairs * 3  # allow for parse failures
        while len(pairs) < n_pairs and attempt < max_attempts:
            idx = order[attempt % len(order)]
            variant = attempt // len(order)
            chunk = chunks[idx]
            attempt += 1

            def _do():
                return self._generate_one(chunk, variant)

            if timing is not None:
                with timing.timed("evalgen_call", arm=self.arm_id):
                    parsed = _do()
            else:
                parsed = _do()

            if parsed is None:
                continue
            pairs.append(QAPair(
                id=f"{self.arm_id}-{len(pairs):05d}",
                arm_id=self.arm_id,
                question=parsed["question"],
                answer=parsed["answer"],
                gold_doc_id=chunk.doc_id,
                gold_chunk_id=chunk.chunk_id,
                source_text=chunk.text,
            ))
        return pairs


def generate_arm_evalset(cfg: Config, arm_spec: Any, chunks: list[Chunk],
                         timing: Optional[TimingLog] = None) -> list[QAPair]:
    """Generate one arm's full synthetic eval set (count from config)."""
    n_pairs = int(cfg.evalgen.get("n_pairs_per_arm", 100))
    gen = EvalGenerator(arm_spec, cache_dir=cfg.runtime.cache_dir, seed=cfg.splits.seed)
    return gen.generate(chunks, n_pairs=n_pairs, timing=timing)
