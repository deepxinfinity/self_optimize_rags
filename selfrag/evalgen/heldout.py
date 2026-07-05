"""Held-out set access guard + real-questions loader.

The credibility of the whole study rests on this: no decision (tuning or selection) may depend on
the synthetic ``test`` split or the real held-out set. We enforce that *in code*, not just by
discipline:

  - A HeldoutSet is created LOCKED. Any access while locked raises — so if optimization code ever
    reads it, the run fails loudly.
  - The runner UNLOCKS held-out sets only after all selection is finished, then evaluates the final
    configs. Every access is recorded, and the log is written to the run for auditability.
"""

from __future__ import annotations

from pathlib import Path

import json

from .generator import QAPair


class HeldoutAccessError(RuntimeError):
    pass


class HeldoutSet:
    def __init__(self, name: str, pairs: list[QAPair]) -> None:
        self.name = name
        self._pairs = pairs
        self._unlocked = False
        self.accesses: list[str] = []

    def unlock(self) -> None:
        self._unlocked = True

    def take(self, by: str) -> list[QAPair]:
        if not self._unlocked:
            raise HeldoutAccessError(
                f"held-out set {self.name!r} accessed by {by!r} during optimization — forbidden. "
                "Only 'val' may drive selection; test/real are for final reporting only."
            )
        self.accesses.append(by)
        return self._pairs

    @property
    def n(self) -> int:
        return len(self._pairs)


def load_real_questions(path: str | Path) -> list[QAPair]:
    """Load data/real_questions.jsonl ({question, answer, gold_doc_id}) as QAPairs."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"real_questions not found: {path}")
    pairs: list[QAPair] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append(QAPair(
                id=f"real-{i:05d}",
                arm_id="real",
                question=obj["question"],
                answer=obj.get("answer", ""),
                gold_doc_id=obj["gold_doc_id"],
                gold_chunk_id="",
                source_text="",
            ))
    if not pairs:
        raise ValueError(f"no real questions found in {path}")
    return pairs
