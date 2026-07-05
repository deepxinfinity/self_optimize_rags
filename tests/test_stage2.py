"""Increment-5 tests: Stage 2 manual-loop logic with injected scorer/proposer (no models)."""

from __future__ import annotations

from selfrag.config import load_config
from selfrag.optimize.stage2_prompts import Stage2Optimizer, accept, select_failures
from selfrag.params import baseline_retrieval_params

CONFIG = "configs/default.yaml"


def test_accept_only_on_strict_improvement():
    assert accept(0.8, 0.7) is True
    assert accept(0.7, 0.7) is False
    assert accept(0.6, 0.7) is False


def test_select_failures_picks_lowest():
    per = [
        {"question": "a", "faithfulness": 0.9},
        {"question": "b", "faithfulness": 0.2},
        {"question": "c", "faithfulness": 0.5},
    ]
    worst = select_failures(per, "faithfulness", k=2)
    assert [f["question"] for f in worst] == ["b", "c"]


def _optimizer_with_stubs(scorer, proposer, n_trials=3):
    cfg = load_config(CONFIG, [f"optimization.stage2_prompts.n_trials={n_trials}",
                               "optimization.stage2_prompts.method=manual"])
    pairs = []  # unused: the injected scorer ignores pairs
    frozen = baseline_retrieval_params(cfg)
    return Stage2Optimizer(cfg, pairs, frozen, scorer=scorer, proposer=proposer)


def test_loop_keeps_best_when_improving():
    # Each proposed prompt scores higher than the last -> best should be the final proposal.
    scores = {"SEED": 0.4, "p1": 0.5, "p2": 0.6, "p3": 0.7}
    proposals = iter(["p1", "p2", "p3"])

    def scorer(prompt):
        return scores[prompt], [{"faithfulness": scores[prompt], "answer_relevancy": 0.0}]

    def proposer(cur, failures):
        return next(proposals)

    opt = _optimizer_with_stubs(scorer, proposer, n_trials=3)
    opt.seed_prompt = "SEED"
    res = opt.run()
    assert res.seed_value == 0.4
    assert res.best_value == 0.7 and res.best_prompt == "p3"


def test_loop_rejects_worse_proposals():
    # All proposals are worse than the seed -> seed must be retained.
    def scorer(prompt):
        v = 0.9 if prompt == "SEED" else 0.3
        return v, [{"faithfulness": v, "answer_relevancy": 0.0}]

    def proposer(cur, failures):
        return "worse"

    opt = _optimizer_with_stubs(scorer, proposer, n_trials=4)
    opt.seed_prompt = "SEED"
    res = opt.run()
    assert res.best_prompt == "SEED" and res.best_value == 0.9
