from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from .generator import QAPair


def split_pairs(pairs: list[QAPair], train: float, val: float, test: float,
                seed: int = 42) -> dict[str, list[QAPair]]:
    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {train + val + test}")
    ordered = sorted(pairs, key=lambda p: p.id)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n = len(ordered)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    splits = {
        "train": ordered[:n_train],
        "val": ordered[n_train:n_train + n_val],
        "test": ordered[n_train + n_val:],
    }
    ids = [p.id for s in splits.values() for p in s]
    assert len(ids) == len(set(ids)) == n, "splits overlap or drop items"
    return splits


def write_jsonl(pairs: Iterable[QAPair], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[QAPair]:
    path = Path(path)
    pairs: list[QAPair] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(QAPair(**json.loads(line)))
    return pairs
