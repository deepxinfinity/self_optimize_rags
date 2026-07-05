"""The self-optimizer: orchestrate both arms through both stages, then evaluate on held-out.

Flow (per the project brief §6-§8):
  1. For each arm: Stage 1 (retrieval, on val) -> freeze -> Stage 2 (answer prompt, on val).
  2. Selection is now complete. UNLOCK the held-out sets (synthetic test + real).
  3. Evaluate the baseline and each arm's winning config on synthetic test and the REAL held-out
     set — exactly once, in this final phase (enforced by HeldoutSet).
  4. Write best_configs.json and print the results table. The real held-out column is the result;
     the gap between arms on it is the synthetic-eval-reliability finding.

The headline metric is the Stage-2 judged metric (faithfulness by default); retrieval metrics and
both judged metrics are also recorded in best_configs.json. Every stage logs per-trial timing.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from ..config import Config
from ..evalgen.heldout import HeldoutSet, load_real_questions
from ..evalgen.splits import load_jsonl
from ..evaluate import evaluate
from ..models.factory import build_judge
from ..params import RetrievalParams, baseline_retrieval_params
from ..rag import build_rag
from ..utils.runs import write_json
from .stage1_retrieval import Stage1Optimizer
from .stage2_prompts import Stage2Optimizer


def _eval_config(cfg: Config, params: RetrievalParams, answer_prompt: str, pairs, judge,
                 sample_size: Optional[int]) -> dict[str, Any]:
    rag = build_rag(cfg, params)
    rag.answerer.answer_prompt = answer_prompt.strip()
    return evaluate(rag, pairs, judge=judge, sample_size=sample_size)


def _fmt(x: Any) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)


def format_table(report: dict[str, Any]) -> str:
    metric = report["headline_metric"]
    rows = [("Setup", "Synth val", "Synth test", "Real held-out")]
    b = report["baseline"]["scores"]
    rows.append(("Baseline (untuned)", "—", _fmt(b["synthetic_test_mean"]), _fmt(b["real_heldout"])))
    for arm_id, arm in report["arms"].items():
        s = arm["scores"]
        rows.append((f"Optimized — {arm_id}", _fmt(s["synthetic_val"]),
                     _fmt(s["synthetic_test"]), _fmt(s["real_heldout"])))
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    line = lambda r: "  ".join(c.ljust(widths[i]) for i, c in enumerate(r))
    out = [f"Headline metric: {metric} (higher is better)", "", line(rows[0]),
           "  ".join("-" * w for w in widths)]
    out += [line(r) for r in rows[1:]]
    return "\n".join(out)


def run_optimize(cfg: Config, run_dir: Path) -> dict[str, Any]:
    out_root = Path(cfg.evalgen.get("out_dir", "data/evalsets"))
    stage2_cfg = cfg.optimization.get("stage2_prompts", {})
    headline = stage2_cfg.get("metric", "faithfulness")
    final_sample = stage2_cfg.get("final_sample_size")  # None => evaluate all held-out
    judge = build_judge(cfg.models.judge, cfg.runtime.cache_dir)
    seed_prompt = cfg.search_space["prompts"]["answer_prompt_seed"]

    # Held-out sets are created LOCKED; any read during optimization raises.
    real_heldout = HeldoutSet("real", load_real_questions(cfg.dataset.real_questions_path))

    arms_out: dict[str, Any] = {}
    test_heldout: dict[str, HeldoutSet] = {}

    # ---- Phase 1: per-arm optimization on val only ----
    for arm in cfg.models.eval_generators:
        val = load_jsonl(out_root / arm.id / "val.jsonl")
        test_heldout[arm.id] = HeldoutSet(f"synthetic_test[{arm.id}]",
                                          load_jsonl(out_root / arm.id / "test.jsonl"))

        s1 = Stage1Optimizer(cfg, val, trial_log_path=run_dir / f"{arm.id}_stage1_trials.jsonl").run()
        s2 = Stage2Optimizer(cfg, val, s1.best_params,
                             trial_log_path=run_dir / f"{arm.id}_stage2_trials.jsonl").run()
        arms_out[arm.id] = {
            "model": arm.name,
            "stage1": {"metric": s1.metric, "baseline": s1.baseline_value, "best": s1.best_value,
                       "n_trials": s1.n_trials, "time_sec": s1.total_seconds},
            "stage2": {"metric": s2.metric, "seed": s2.seed_value, "best": s2.best_value,
                       "n_trials": s2.n_trials, "time_sec": s2.total_seconds},
            "best_config": {"retrieval": asdict(s1.best_params), "answer_prompt": s2.best_prompt},
            "scores": {"synthetic_val": s2.best_value},  # selection score
            "_frozen": s1.best_params, "_prompt": s2.best_prompt,
        }

    # ---- Selection complete: unlock held-out sets (touch once, final reporting only) ----
    real_heldout.unlock()
    for h in test_heldout.values():
        h.unlock()

    baseline_params = baseline_retrieval_params(cfg)
    baseline_test_scores: list[float] = []

    # ---- Phase 2: final one-touch evaluation ----
    for arm_id, arm in arms_out.items():
        frozen, prompt = arm.pop("_frozen"), arm.pop("_prompt")

        test_pairs = test_heldout[arm_id].take(by=f"final_eval:{arm_id}")
        test_rep = _eval_config(cfg, frozen, prompt, test_pairs, judge, final_sample)
        real_rep = _eval_config(cfg, frozen, prompt, real_heldout.take(by=f"optimized:{arm_id}"),
                                judge, final_sample)
        arm["scores"]["synthetic_test"] = test_rep["judged"][headline]
        arm["scores"]["real_heldout"] = real_rep["judged"][headline]
        arm["scores"]["synthetic_test_full"] = test_rep
        arm["scores"]["real_heldout_full"] = real_rep

        # Baseline (untuned) on this arm's synthetic test, for the comparison.
        base_test = _eval_config(cfg, baseline_params, seed_prompt, test_pairs, judge, final_sample)
        baseline_test_scores.append(base_test["judged"][headline])

    base_real = _eval_config(cfg, baseline_params, seed_prompt,
                             real_heldout.take(by="baseline"), judge, final_sample)

    report = {
        "headline_metric": headline,
        "corpus_path": cfg.dataset.corpus_path,
        "arms": arms_out,
        "baseline": {
            "config": {"retrieval": asdict(baseline_params), "answer_prompt": seed_prompt.strip()},
            "scores": {
                "synthetic_test_per_arm": dict(zip(arms_out.keys(), baseline_test_scores)),
                "synthetic_test_mean": round(sum(baseline_test_scores) / len(baseline_test_scores), 4),
                "real_heldout": base_real["judged"][headline],
                "real_heldout_full": base_real,
            },
        },
        "heldout_access_log": {
            "real": real_heldout.accesses,
            **{k: v.accesses for k, v in test_heldout.items()},
        },
    }
    write_json(run_dir / "best_configs.json", report)
    return report
