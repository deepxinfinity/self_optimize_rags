"""Held-out sets are locked until selection is over.

The runner unlocks the synthetic test split and the real questions only after both stages have
picked their configs, so a read from inside the optimizers fails loudly. Every read is logged
and the log is written into best_configs.json.
"""

from __future__ import annotations

import json
from pathlib import Path

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
                f"held-out set {self.name!r} accessed by {by!r} during optimization. "
                "Only val may drive selection; test and real are for the final report."
            )
        self.accesses.append(by)
        return self._pairs

    @property
    def n(self) -> int:
        return len(self._pairs)


def load_real_questions(path: str | Path) -> list[QAPair]:
    """Lines of {question, answer?, gold_doc_id, evidence?}."""
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
                evidence=list(obj.get("evidence") or []),
            ))
    if not pairs:
        raise ValueError(f"no real questions found in {path}")
    return pairs
