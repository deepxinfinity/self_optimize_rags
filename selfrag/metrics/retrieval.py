"""Retrieval-only metrics — cheap, no answer generation.

Scored at the *document* level: a retrieved chunk counts as relevant if its ``doc_id`` matches the
query's ``gold_doc_id``. Doc-level (rather than chunk-level) is what makes these metrics stable
across the Stage-1 ``chunk_size`` sweep, and it matches how the eval sets record gold truth.

These drive Stage 1 of the optimizer (Increment 4): fast enough for many Optuna trials.
"""

from __future__ import annotations


def hit(retrieved_doc_ids: list[str], gold_doc_id: str) -> bool:
    """Did any retrieved chunk come from the gold document?"""
    return gold_doc_id in retrieved_doc_ids


def reciprocal_rank(retrieved_doc_ids: list[str], gold_doc_id: str) -> float:
    """1/rank of the first retrieved chunk from the gold document (0 if none)."""
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == gold_doc_id:
            return 1.0 / rank
    return 0.0


def retrieval_metrics(samples: list[tuple[list[str], str]]) -> dict[str, float]:
    """Aggregate over queries.

    ``samples``: list of (retrieved_doc_ids_in_rank_order, gold_doc_id).
    Returns ``{"hit_rate", "mrr", "n"}``.
    """
    if not samples:
        return {"hit_rate": 0.0, "mrr": 0.0, "n": 0}
    n = len(samples)
    hits = sum(hit(ids, gold) for ids, gold in samples)
    rr = sum(reciprocal_rank(ids, gold) for ids, gold in samples)
    return {"hit_rate": round(hits / n, 4), "mrr": round(rr / n, 4), "n": n}
