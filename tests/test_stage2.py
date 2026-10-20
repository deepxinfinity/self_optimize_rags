from __future__ import annotations

import json

from selfrag.config import load_config
from selfrag.evalgen.generator import QAPair
from selfrag.optimize.stage2_prompts import Stage2Optimizer, accept, select_failures
from selfrag.params import baseline_retrieval_params

CONFIG = "configs/qasper.yaml"
METRIC = "answer_correctness"


def _pairs(prefix: str, n: int) -> list[QAPair]:
    return [QAPair(id=f"{prefix}{i}", arm_id="a", question=f"{prefix}-q{i}", answer="ref",
                   gold_doc_id="d.txt") for i in range(n)]


def _optimizer(scorer, proposer, n_trials=3, min_delta=0.0, train=True, log=None):
    cfg = load_config(CONFIG, [f"optimization.stage2_prompts.n_trials={n_trials}",
                               f"optimization.stage2_prompts.min_delta={min_delta}"])
    opt = Stage2Optimizer(cfg, _pairs("val", 4), baseline_retrieval_params(cfg),
                          train_pairs=_pairs("train", 4) if train else None,
                          trial_log_path=log, scorer=scorer, proposer=proposer)
    opt.seed_prompt = "SEED"
    return opt


def _items(value, pairs):
    return [{"question": p.question, "answer": "a", "reference": p.answer, METRIC: value}
            for p in pairs]


def test_accept_needs_more_than_min_delta():
    assert accept(0.8, 0.7) is True
    assert accept(0.7, 0.7) is False
    assert accept(0.6, 0.7) is False
    assert accept(0.72, 0.7, min_delta=0.05) is False
    assert accept(0.76, 0.7, min_delta=0.05) is True


def test_select_failures_picks_lowest_and_skips_perfect():
    per = [{"question": "a", METRIC: 1.0}, {"question": "b", METRIC: 0.0},
           {"question": "c", METRIC: 0.5}, {"question": "d", METRIC: None}]
    assert [f["question"] for f in select_failures(per, METRIC, k=2)] == ["b", "d"]
    assert [f["question"] for f in select_failures(per, METRIC, k=2, offset=2)] == ["c", "b"]
    assert select_failures([{"question": "a", METRIC: 1.0}], METRIC) == []


def test_loop_keeps_best_when_improving():
    scores = {"SEED": 0.4, "p1": 0.5, "p2": 0.6, "p3": 0.7}
    proposals = iter(["p1", "p2", "p3"])

    def scorer(prompt, pairs):
        return scores[prompt], _items(scores[prompt], pairs)

    res = _optimizer(scorer, lambda cur, failures: next(proposals)).run()
    assert res.seed_value == 0.4
    assert res.best_value == 0.7 and res.best_prompt == "p3"


def test_loop_rejects_worse_proposals():
    def scorer(prompt, pairs):
        v = 0.9 if prompt == "SEED" else 0.3
        return v, _items(v, pairs)

    res = _optimizer(scorer, lambda cur, failures: f"worse-{len(failures)}-{id(failures)}",
                     n_trials=4).run()
    assert res.best_prompt == "SEED" and res.best_value == 0.9


def test_small_gains_below_min_delta_are_rejected():
    def scorer(prompt, pairs):
        v = 0.50 if prompt == "SEED" else 0.51
        return v, _items(v, pairs)

    res = _optimizer(scorer, lambda cur, f: "slightly-better", min_delta=0.02).run()
    assert res.best_prompt == "SEED"


def test_failures_come_from_train_and_acceptance_from_val():
    seen_by_scorer: list[str] = []
    shown_to_proposer: list[str] = []

    def scorer(prompt, pairs):
        seen_by_scorer.append(pairs[0].id[:3])
        return 0.5, _items(0.5, pairs)

    def proposer(cur, failures):
        shown_to_proposer.extend(f["question"] for f in failures)
        return "candidate"

    _optimizer(scorer, proposer, n_trials=1).run()
    assert all(q.startswith("train") for q in shown_to_proposer)
    assert seen_by_scorer == ["val", "tra", "val"]


def test_duplicate_proposals_are_logged_but_not_rescored(tmp_path):
    calls = []

    def scorer(prompt, pairs):
        calls.append(prompt)
        return 0.5, _items(0.5, pairs)

    log = tmp_path / "s2.jsonl"
    res = _optimizer(scorer, lambda cur, f: "same", n_trials=3, log=log).run()
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls.count("same") == 1
    assert [r.get("note") for r in rows] == [None, None, "duplicate", "duplicate"]
    assert res.n_trials == 3


def test_stops_when_nothing_fails():
    def scorer(prompt, pairs):
        return 1.0, _items(1.0, pairs)

    res = _optimizer(scorer, lambda cur, f: "never", n_trials=5).run()
    assert res.n_trials == 0 and res.best_prompt == "SEED"
