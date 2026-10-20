from __future__ import annotations

import math

import pytest

from selfrag.analysis.paired import (
    RETRIEVAL_TESTS,
    compare_records,
    format_comparison,
    mcnemar,
    paired_bootstrap,
    wilcoxon_signed_rank,
)


def test_mcnemar_exact_known_value():
    # 5 discordant pairs all favouring a: p = 2 * 0.5^5
    a = [True] * 5 + [True, False]
    b = [False] * 5 + [True, False]
    res = mcnemar(a, b)
    assert res.detail["a_only"] == 5
    assert res.detail["b_only"] == 0
    assert res.p_value == pytest.approx(2 * 0.5 ** 5)
    assert not res.significant


def test_mcnemar_exact_significant_at_six():
    a = [True] * 6
    b = [False] * 6
    res = mcnemar(a, b)
    assert res.p_value == pytest.approx(2 * 0.5 ** 6)
    assert res.significant


def test_mcnemar_ignores_concordant_pairs():
    # concordant pairs change the means but not the p-value
    a_small, b_small = [True] * 6, [False] * 6
    a_big = a_small + [True] * 50 + [False] * 50
    b_big = b_small + [True] * 50 + [False] * 50
    assert mcnemar(a_big, b_big).p_value == pytest.approx(mcnemar(a_small, b_small).p_value)
    assert mcnemar(a_big, b_big).delta != pytest.approx(mcnemar(a_small, b_small).delta)


def test_mcnemar_no_discordant_pairs():
    res = mcnemar([True, False], [True, False])
    assert res.p_value == 1.0
    assert res.delta == 0.0


def test_mcnemar_symmetric_under_swap():
    a = [True] * 7 + [False] * 3
    b = [False] * 7 + [True] * 3
    assert mcnemar(a, b).p_value == pytest.approx(mcnemar(b, a).p_value)


def test_wilcoxon_all_zero_differences():
    res = wilcoxon_signed_rank([0.5] * 10, [0.5] * 10)
    assert res.p_value == 1.0
    assert res.n == 0


def test_wilcoxon_drops_zero_differences():
    a = [1.0, 0.5, 0.25] + [0.5] * 20
    b = [0.5, 0.25, 0.0] + [0.5] * 20
    res = wilcoxon_signed_rank(a, b)
    assert res.n == 3
    assert res.detail["n_dropped_zero"] == 20


def test_wilcoxon_detects_consistent_improvement():
    a = [0.5] * 20
    b = [0.1 + 0.01 * i for i in range(20)]
    res = wilcoxon_signed_rank(a, b)
    assert res.delta > 0
    assert res.significant


def test_wilcoxon_symmetric_differences_are_not_significant():
    a = [1.0, 0.0] * 10
    b = [0.0, 1.0] * 10
    res = wilcoxon_signed_rank(a, b)
    assert res.delta == pytest.approx(0.0)
    assert not res.significant


@pytest.mark.parametrize("n", [8, 15, 30])
def test_wilcoxon_matches_scipy_when_available(n):
    scipy_stats = pytest.importorskip("scipy.stats")
    # rr-like values with ties in |diff|
    a = [(i % 4) / 4.0 for i in range(n)]
    b = [((i + 1) % 5) / 5.0 for i in range(n)]
    ours = wilcoxon_signed_rank(a, b)
    theirs = scipy_stats.wilcoxon(a, b, zero_method="wilcox", correction=True,
                                  mode="approx" if hasattr(scipy_stats, "wilcoxon") else "auto")
    assert ours.p_value == pytest.approx(float(theirs.pvalue), abs=1e-6)


def test_wilcoxon_rejects_misaligned_input():
    with pytest.raises(ValueError):
        wilcoxon_signed_rank([1.0, 2.0], [1.0])


def test_bootstrap_ci_brackets_the_delta():
    a = [0.6] * 40
    b = [0.4] * 40
    res = paired_bootstrap(a, b, n_boot=2000)
    assert res.delta == pytest.approx(0.2)
    assert res.ci_low == pytest.approx(0.2)
    assert res.ci_high == pytest.approx(0.2)


def test_bootstrap_is_deterministic_for_a_seed():
    a = [i / 20 for i in range(20)]
    b = [(i % 3) / 3 for i in range(20)]
    first = paired_bootstrap(a, b, n_boot=500, seed=7)
    second = paired_bootstrap(a, b, n_boot=500, seed=7)
    assert first.p_value == second.p_value
    assert first.ci_low == second.ci_low


def test_bootstrap_counts_direction():
    a = [1.0, 1.0, 0.0, 0.5]
    b = [0.0, 0.5, 1.0, 0.5]
    res = paired_bootstrap(a, b, n_boot=500)
    assert res.detail["n_improved"] == 2
    assert res.detail["n_worsened"] == 1
    assert res.detail["n_tied"] == 1


def test_bootstrap_no_difference_is_not_significant():
    a = [0.0, 0.5, 1.0] * 10
    b = [0.5, 1.0, 0.0] * 10
    res = paired_bootstrap(a, b, n_boot=2000)
    assert not res.significant


def test_bootstrap_rejects_empty():
    with pytest.raises(ValueError):
        paired_bootstrap([], [])


def _records(rrs: list[float], start: int = 0) -> list[dict]:
    return [{"id": f"real-{start + i:05d}", "question": f"q{i}", "rr": rr, "hit": rr > 0}
            for i, rr in enumerate(rrs)]


def test_compare_records_aligns_by_id_not_order():
    a = _records([1.0, 0.5, 0.0])
    b = _records([0.5, 0.25, 0.0])
    straight = compare_records(a, b, RETRIEVAL_TESTS)
    shuffled = compare_records(a, [b[2], b[0], b[1]], RETRIEVAL_TESTS)
    assert straight["metrics"]["rr"]["bootstrap"]["delta"] == pytest.approx(
        shuffled["metrics"]["rr"]["bootstrap"]["delta"])
    assert straight["n_compared"] == 3


def test_compare_records_reports_unmatched():
    comp = compare_records(_records([1.0, 0.5, 0.25]), _records([1.0, 0.5]), RETRIEVAL_TESTS)
    assert comp["n_compared"] == 2
    assert comp["n_dropped_unmatched"] == 1


def test_compare_records_requires_overlap():
    with pytest.raises(ValueError):
        compare_records(_records([1.0], start=0), _records([1.0], start=99), RETRIEVAL_TESTS)


def test_compare_records_picks_the_test_per_metric_kind():
    a = _records([1.0] * 12 + [0.0] * 8)
    b = _records([0.5] * 12 + [0.0] * 8)
    comp = compare_records(a, b, RETRIEVAL_TESTS)
    assert set(comp["metrics"]) == {"rr", "hit"}     # passage fields absent from the records
    assert comp["metrics"]["rr"]["bootstrap"]["delta"] == pytest.approx(0.3)
    assert comp["metrics"]["rr"]["wilcoxon"]["significant"]
    assert comp["metrics"]["hit"]["mcnemar"]["p_value"] == 1.0


def test_compare_records_skips_missing_values():
    a = [{"id": "1", "answer_correctness": 1.0}, {"id": "2", "answer_correctness": None}]
    b = [{"id": "1", "answer_correctness": 0.5}, {"id": "2", "answer_correctness": 0.5}]
    comp = compare_records(a, b, {"answer_correctness": "continuous"})
    assert comp["metrics"]["answer_correctness"]["bootstrap"]["n"] == 1


def test_format_comparison_mentions_every_metric():
    comp = compare_records(_records([1.0] * 12), _records([0.5] * 12), RETRIEVAL_TESTS)
    text = "\n".join(format_comparison("arm vs baseline", comp))
    assert "rr" in text and "hit" in text and "McNemar" in text


def test_paired_beats_unpaired_on_a_shared_variance_case():
    """A small consistent gain under large between-question variance: paired finds it,
    an unpaired z-test on the same numbers does not."""
    n = 250
    difficulty = [(i % 7) / 7.0 for i in range(n)]
    b = difficulty
    a = [min(1.0, d + 0.03) for d in difficulty]

    assert paired_bootstrap(a, b, n_boot=2000).significant

    mean_a, mean_b = sum(a) / n, sum(b) / n
    var_a = sum((x - mean_a) ** 2 for x in a) / (n - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n - 1)
    z = (mean_a - mean_b) / math.sqrt(var_a / n + var_b / n)
    assert abs(z) < 1.96
