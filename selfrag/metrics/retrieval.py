"""Retrieval metrics at two granularities.

Document level: did a chunk from the gold document come back (hit / rr).
Passage level: did a chunk that actually contains the gold evidence come back (passage_hit /
passage_rr). A chunk counts as containing a piece of evidence when at least half of the 5-grams
of the shorter of the two are shared, which works whether the evidence is a sentence inside a
large chunk or a long paragraph split across several small chunks.
"""

from __future__ import annotations

import re
from functools import lru_cache
from statistics import mean
from typing import Any, Sequence

from ..corpus.chunker import Chunk

_WORD = re.compile(r"\w+")
NGRAM = 5
RELEVANCE_THRESHOLD = 0.5


def ngrams(text: str, n: int = NGRAM) -> frozenset[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return frozenset([tuple(words)]) if words else frozenset()
    return frozenset(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


@lru_cache(maxsize=8192)
def _evidence_ngrams(text: str) -> frozenset[tuple[str, ...]]:
    return ngrams(text)


def overlap(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def passage_relevant(text: str, evidence: Sequence[str],
                     threshold: float = RELEVANCE_THRESHOLD) -> bool:
    grams = ngrams(text)
    return any(overlap(grams, _evidence_ngrams(ev)) >= threshold for ev in evidence)


def hit(retrieved_doc_ids: list[str], gold_doc_id: str) -> bool:
    return gold_doc_id in retrieved_doc_ids


def reciprocal_rank(retrieved_doc_ids: list[str], gold_doc_id: str) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == gold_doc_id:
            return 1.0 / rank
    return 0.0


def retrieval_record(chunks: list[Chunk], gold_doc_id: str,
                     evidence: Sequence[str] = ()) -> dict[str, Any]:
    doc_ids = [c.doc_id for c in chunks]
    rr = reciprocal_rank(doc_ids, gold_doc_id)
    rec: dict[str, Any] = {
        "gold_doc_id": gold_doc_id,
        "retrieved": [c.chunk_id for c in chunks],
        "hit": rr > 0,
        "rr": rr,
    }
    if evidence:
        rank = next((i for i, c in enumerate(chunks, start=1)
                     if c.doc_id == gold_doc_id and passage_relevant(c.text, evidence)), None)
        rec["passage_hit"] = rank is not None
        rec["passage_rr"] = 1.0 / rank if rank else 0.0
    return rec


def summarize_retrieval(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"hit_rate": 0.0, "mrr": 0.0, "n": 0}
    out: dict[str, Any] = {
        "hit_rate": round(mean(float(r["hit"]) for r in records), 4),
        "mrr": round(mean(r["rr"] for r in records), 4),
        "n": len(records),
    }
    with_evidence = [r for r in records if "passage_hit" in r]
    if with_evidence:
        out["passage_hit_rate"] = round(mean(float(r["passage_hit"]) for r in with_evidence), 4)
        out["passage_mrr"] = round(mean(r["passage_rr"] for r in with_evidence), 4)
        out["n_with_evidence"] = len(with_evidence)
    return out
