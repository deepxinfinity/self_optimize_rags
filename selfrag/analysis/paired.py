"""Paired significance tests for two configs scored on the same questions.

Per-question differences cancel out question difficulty, which dominates the variance here, so a
paired test finds effects that comparing two means would miss. numpy only; the Wilcoxon normal
approximation is checked against scipy in the tests.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import numpy as np

RETRIEVAL_TESTS = {"passage_hit": "binary", "passage_rr": "continuous",
                   "hit": "binary", "rr": "continuous"}

SIG_NOTE = ("SIG = p < 0.05 (for continuous metrics, bootstrap and Wilcoxon both). "
            "ns with a CI spanning zero means underpowered, not no effect.")


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class PairedResult:
    test: str
    metric: str
    n: int
    mean_a: float
    mean_b: float
    delta: float
    p_value: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    detail: Optional[dict] = None

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def to_dict(self) -> dict:
        d = asdict(self)
        d["significant"] = self.significant
        return d


def _check(a: Sequence[float], b: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"paired samples must align: got {a_arr.shape} vs {b_arr.shape}")
    if a_arr.size == 0:
        raise ValueError("paired samples are empty")
    return a_arr, b_arr


def paired_bootstrap(a: Sequence[float], b: Sequence[float], metric: str = "value",
                     n_boot: int = 10000, seed: int = 42,
                     alpha: float = 0.05) -> PairedResult:
    """CI for mean(a - b), resampling questions so the pairing is kept."""
    a_arr, b_arr = _check(a, b)
    diff = a_arr - b_arr
    n = diff.size
    rng = np.random.default_rng(seed)
    boot_means = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    p_left = float(np.mean(boot_means <= 0.0))
    p_right = float(np.mean(boot_means >= 0.0))
    return PairedResult(
        test="paired_bootstrap", metric=metric, n=n,
        mean_a=float(a_arr.mean()), mean_b=float(b_arr.mean()), delta=float(diff.mean()),
        p_value=min(1.0, 2.0 * min(p_left, p_right)), ci_low=float(lo), ci_high=float(hi),
        detail={"n_boot": n_boot, "seed": seed,
                "n_improved": int(np.sum(diff > 0)),
                "n_worsened": int(np.sum(diff < 0)),
                "n_tied": int(np.sum(diff == 0))},
    )


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float],
                         metric: str = "value") -> PairedResult:
    """Two-sided, normal approximation with tie and continuity correction; zero diffs dropped."""
    a_arr, b_arr = _check(a, b)
    diff = a_arr - b_arr
    nonzero = diff[diff != 0.0]
    n = nonzero.size
    base = dict(test="wilcoxon_signed_rank", metric=metric,
                mean_a=float(a_arr.mean()), mean_b=float(b_arr.mean()),
                delta=float(diff.mean()))
    if n == 0:
        return PairedResult(n=0, p_value=1.0, detail={"note": "all differences are zero"}, **base)

    abs_d = np.abs(nonzero)
    order = np.argsort(abs_d, kind="mergesort")
    sorted_abs = abs_d[order]
    ranks = np.empty(n, dtype=float)
    tie_groups: list[int] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        tie_groups.append(j - i + 1)
        i = j + 1

    w_plus = float(ranks[nonzero > 0].sum())
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - sum(t ** 3 - t for t in tie_groups) / 48.0
    if var_w <= 0:
        return PairedResult(n=n, p_value=1.0,
                            detail={"note": "zero variance after tie correction"}, **base)
    z = w_plus - mean_w
    z = (z - math.copysign(0.5, z)) / math.sqrt(var_w) if z != 0 else 0.0
    return PairedResult(
        n=n, p_value=min(1.0, 2.0 * (1.0 - _phi(abs(z)))),
        detail={"w_plus": w_plus, "z": z, "n_dropped_zero": int(diff.size - n),
                "n_improved": int(np.sum(nonzero > 0)),
                "n_worsened": int(np.sum(nonzero < 0))},
        **base,
    )


def mcnemar(a: Sequence[bool], b: Sequence[bool], metric: str = "hit") -> PairedResult:
    """Exact two-sided McNemar test. Only questions where the configs disagree count."""
    a_arr = np.asarray(a, dtype=bool)
    b_arr = np.asarray(b, dtype=bool)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"paired samples must align: got {a_arr.shape} vs {b_arr.shape}")
    if a_arr.size == 0:
        raise ValueError("paired samples are empty")
    a_only = int(np.sum(a_arr & ~b_arr))
    b_only = int(np.sum(~a_arr & b_arr))
    n_disc = a_only + b_only
    base = dict(test="mcnemar_exact", metric=metric, n=int(a_arr.size),
                mean_a=float(a_arr.mean()), mean_b=float(b_arr.mean()),
                delta=float(a_arr.mean() - b_arr.mean()))
    detail = {"a_only": a_only, "b_only": b_only, "n_discordant": n_disc}
    if n_disc == 0:
        return PairedResult(p_value=1.0, detail=detail, **base)
    tail = sum(math.comb(n_disc, i) for i in range(min(a_only, b_only) + 1)) * 0.5 ** n_disc
    return PairedResult(p_value=min(1.0, 2.0 * tail), detail=detail, **base)


def compare_records(a_records: list[dict], b_records: list[dict], tests: dict[str, str],
                    seed: int = 42) -> dict[str, Any]:
    """Run each metric's test on two per-question dumps, aligned by question id.

    `tests` maps a record field to "binary" (McNemar) or "continuous" (bootstrap + Wilcoxon).
    Fields missing from either side are skipped; unmatched questions are counted, never imputed.
    """
    a_by_id = {r["id"]: r for r in a_records}
    b_by_id = {r["id"]: r for r in b_records}
    common = sorted(qid for qid in a_by_id if qid in b_by_id)
    if not common:
        raise ValueError("no overlapping question ids between the two dumps")
    out: dict[str, Any] = {"n_compared": len(common),
                           "n_dropped_unmatched": len(a_by_id) + len(b_by_id) - 2 * len(common),
                           "metrics": {}}
    for field, kind in tests.items():
        pairs = [(a_by_id[q].get(field), b_by_id[q].get(field)) for q in common]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if not pairs:
            continue
        a, b = zip(*pairs)
        if kind == "binary":
            out["metrics"][field] = {"mcnemar": mcnemar(a, b, metric=field).to_dict()}
        else:
            out["metrics"][field] = {
                "bootstrap": paired_bootstrap(a, b, metric=field, seed=seed).to_dict(),
                "wilcoxon": wilcoxon_signed_rank(a, b, metric=field).to_dict(),
            }
    return out


def format_comparison(title: str, comp: dict[str, Any]) -> list[str]:
    lines = [f"{title}  (n={comp['n_compared']})"]
    for field, tests in comp["metrics"].items():
        if "mcnemar" in tests:
            t = tests["mcnemar"]
            d = t["detail"]
            lines.append(f"  {field:<20}{t['delta']:+.4f}  p={t['p_value']:.4f} "
                         f"{'SIG' if t['significant'] else 'ns '}  "
                         f"(McNemar: {d['a_only']} won, {d['b_only']} lost)")
        else:
            bt, wx = tests["bootstrap"], tests["wilcoxon"]
            # the bootstrap p is optimistic on small n, so both tests have to agree
            sig = bt["significant"] and wx["significant"]
            lines.append(f"  {field:<20}{bt['delta']:+.4f}  "
                         f"95% CI [{bt['ci_low']:+.4f}, {bt['ci_high']:+.4f}]  "
                         f"p={bt['p_value']:.4f} {'SIG' if sig else 'ns '}  "
                         f"(Wilcoxon p={wx['p_value']:.4f})")
    return lines
