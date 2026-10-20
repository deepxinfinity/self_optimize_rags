"""Full optimize run: per arm Stage 1 then Stage 2 on val, then one final pass over held-out data.

Held-out sets stay locked until every arm has picked its config. The final pass scores the
baseline and each winner on the arm's synthetic test split and on the real questions, dumps
per-question results, and runs paired tests on them.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from ..analysis.paired import RETRIEVAL_TESTS, SIG_NOTE, compare_records, format_comparison
from ..config import Config
from ..evalgen.generator import QAPair
from ..evalgen.heldout import HeldoutSet, load_real_questions
from ..evalgen.splits import load_jsonl
from ..evaluate import evaluate
from ..metrics.judged import DEFAULT_METRICS
from ..models.factory import build_judge
from ..params import RetrievalParams, baseline_retrieval_params, describe, params_from_dict
from ..rag import RagFactory
from ..utils.runs import write_json, write_jsonl
from ..utils.timing import TimingLog
from .stage1_retrieval import Stage1Optimizer
from .stage2_prompts import Stage2Optimizer


def _judged_metrics(cfg: Config, headline: str) -> tuple[str, ...]:
    metrics = tuple(cfg.optimization.get("judged_metrics") or DEFAULT_METRICS)
    return metrics if headline in metrics else (headline, *metrics)


class _FinalEval:
    def __init__(self, factory: RagFactory, judge: Any, metrics: tuple[str, ...],
                 run_dir: Path, log: Callable[[str], None] = print) -> None:
        self.log = log
        self.factory = factory
        self.judge = judge
        self.metrics = metrics
        self.run_dir = run_dir
        self.per_question: dict[str, dict[str, list[dict]]] = {}

    def __call__(self, label: str, params: RetrievalParams, prompt: str,
                 pairs: list[QAPair], judged: Optional[int]) -> dict[str, Any]:
        rag = self.factory.build(params, prompt)
        n_judged = min(judged, len(pairs)) if judged else len(pairs)
        self.log(f"final eval: {label} (retrieval on {len(pairs)}, judging {n_judged})")
        with self.factory.timing.timed("final_eval", config=label, n_pairs=len(pairs)):
            report = evaluate(rag, pairs, judge=self.judge, metrics=self.metrics,
                              sample_size=judged, timing=self.factory.timing, log=self.log)
        per_q = report.pop("per_question")
        self.per_question[label] = per_q
        for kind, records in per_q.items():
            write_jsonl(self.run_dir / "per_question" / f"{label}__{kind}.jsonl", records)
        return report

    def compare(self, a: str, b: str, headline: str) -> dict[str, Any]:
        pa, pb = self.per_question[a], self.per_question[b]
        out = {"retrieval": compare_records(pa["retrieval"], pb["retrieval"], RETRIEVAL_TESTS),
               "judged": {"metrics": {}}}
        if pa.get("judged") and pb.get("judged"):
            tests = {m: "continuous" for m in (headline, "faithfulness", "answer_f1")}
            out["judged"] = compare_records(pa["judged"], pb["judged"], tests)
        return out

    def compare_real(self, arm_ids: list[str], headline: str) -> dict[str, Any]:
        out = {f"{a}_vs_baseline": self.compare(f"{a}__real_heldout", "baseline__real_heldout",
                                                headline)
               for a in arm_ids}
        for i, a in enumerate(arm_ids):
            for b in arm_ids[i + 1:]:
                out[f"{a}_vs_{b}"] = self.compare(f"{a}__real_heldout", f"{b}__real_heldout",
                                                  headline)
        return out


def run_optimize(cfg: Config, run_dir: Path,
                 log: Callable[[str], None] = print) -> dict[str, Any]:
    out_root = Path(cfg.evalgen.get("out_dir", "data/evalsets"))
    stage2_cfg = cfg.optimization.get("stage2_prompts", {})
    headline = stage2_cfg.get("metric", "answer_correctness")
    # retrieval metrics always cover every held-out pair; these cap the judged answers
    final_cfg = cfg.optimization.get("final_eval", {})
    real_n = final_cfg.get("real_sample_size")
    synth_n = final_cfg.get("synthetic_sample_size")
    seed_prompt = cfg.search_space["prompts"]["answer_prompt_seed"].strip()

    timing = TimingLog()
    factory = RagFactory(cfg, timing)
    judge = build_judge(cfg.models.judge, cfg.runtime.cache_dir)

    real: Optional[HeldoutSet] = None
    if cfg.dataset.real_questions_path:
        real = HeldoutSet("real", load_real_questions(cfg.dataset.real_questions_path))

    arms_out: dict[str, Any] = {}
    tests: dict[str, HeldoutSet] = {}
    winners: dict[str, tuple[RetrievalParams, str]] = {}

    for arm in cfg.models.eval_generators:
        arm_dir = out_root / arm.id
        val = load_jsonl(arm_dir / "val.jsonl")
        train = load_jsonl(arm_dir / "train.jsonl")
        tests[arm.id] = HeldoutSet(f"synthetic_test[{arm.id}]", load_jsonl(arm_dir / "test.jsonl"))

        log(f"{arm.id}: stage 1 on {len(val)} val pairs")
        s1 = Stage1Optimizer(cfg, val, trial_log_path=run_dir / f"{arm.id}_stage1_trials.jsonl",
                             factory=factory, log=log).run()
        log(f"{arm.id}: stage 1 {s1.metric} {s1.baseline_value} -> {s1.best_value}")
        log(f"{arm.id}: stage 2 (answer prompt), retrieval frozen at {describe(s1.best_params)}")
        s2 = Stage2Optimizer(cfg, val, s1.best_params, train_pairs=train,
                             trial_log_path=run_dir / f"{arm.id}_stage2_trials.jsonl",
                             factory=factory, judge=judge, log=log).run()
        log(f"{arm.id}: stage 2 {s2.metric} {s2.seed_value} -> {s2.best_value}")

        winners[arm.id] = (s1.best_params, s2.best_prompt)
        arms_out[arm.id] = {
            "model": arm.name,
            "stage1": {"metric": s1.metric, "baseline": s1.baseline_value, "best": s1.best_value,
                       "best_metrics": s1.best_metrics, "n_trials": s1.n_trials,
                       "time_sec": s1.total_seconds},
            "stage2": {"metric": s2.metric, "seed": s2.seed_value, "best": s2.best_value,
                       "n_trials": s2.n_trials, "time_sec": s2.total_seconds},
            "best_config": {"retrieval": asdict(s1.best_params), "answer_prompt": s2.best_prompt},
            "scores": {"synthetic_val": s2.best_value},
        }

    # selection is over; everything below only reports
    for h in tests.values():
        h.unlock()
    if real is not None:
        real.unlock()

    metrics = _judged_metrics(cfg, headline)
    final = _FinalEval(factory, judge, metrics, run_dir, log=log)
    baseline_params = baseline_retrieval_params(cfg)
    if real is not None:
        judged_n = min(real_n, real.n) if real_n else real.n
        log(f"final eval: judging {judged_n}/{real.n} real questions")

    significance: dict[str, Any] = {"synthetic_test": {}, "real_heldout": {}}
    baseline_test: dict[str, float] = {}
    for arm_id, (params, prompt) in winners.items():
        scores = arms_out[arm_id]["scores"]
        test_pairs = tests[arm_id].take(by=f"final_eval:{arm_id}")
        rep = final(f"{arm_id}__synthetic_test", params, prompt, test_pairs, synth_n)
        scores["synthetic_test"] = rep["judged"].get(headline)
        scores["synthetic_test_full"] = rep
        base = final(f"baseline__synthetic_test__{arm_id}", baseline_params, seed_prompt,
                     test_pairs, synth_n)
        baseline_test[arm_id] = base["judged"].get(headline)
        significance["synthetic_test"][f"{arm_id}_vs_baseline"] = final.compare(
            f"{arm_id}__synthetic_test", f"baseline__synthetic_test__{arm_id}", headline)

        scores["real_heldout"] = None
        if real is not None:
            rep = final(f"{arm_id}__real_heldout", params, prompt,
                        real.take(by=f"optimized:{arm_id}"), real_n)
            scores["real_heldout"] = rep["judged"].get(headline)
            scores["real_heldout_full"] = rep

    valid_base = [v for v in baseline_test.values() if v is not None]
    baseline_scores: dict[str, Any] = {
        "synthetic_test_per_arm": baseline_test,
        "synthetic_test_mean": round(sum(valid_base) / len(valid_base), 4) if valid_base else None,
        "real_heldout": None,
    }
    access_log: dict[str, Any] = {k: v.accesses for k, v in tests.items()}
    if real is not None:
        base_real = final("baseline__real_heldout", baseline_params, seed_prompt,
                          real.take(by="baseline"), real_n)
        baseline_scores["real_heldout"] = base_real["judged"].get(headline)
        baseline_scores["real_heldout_full"] = base_real
        access_log["real"] = real.accesses
        significance["real_heldout"] = final.compare_real(list(winners), headline)

    report = {
        "headline_metric": headline,
        "judged_metrics": list(metrics),
        "corpus_path": cfg.dataset.corpus_path,
        "has_real_heldout": real is not None,
        "arms": arms_out,
        "baseline": {
            "config": {"retrieval": asdict(baseline_params), "answer_prompt": seed_prompt},
            "scores": baseline_scores,
        },
        "significance": significance,
        "final_eval": {
            "real_judged": real_n,
            "synthetic_judged": synth_n,
            "context_budget_words": factory.context_budget,
            "indices_built": factory.n_indices_built,
        },
        "heldout_access_log": access_log,
    }
    write_json(run_dir / "best_configs.json", report)
    timing.dump(run_dir / "timings.json")
    return report


def run_reeval(cfg: Config, source_run: Path, out_dir: Path,
               log: Callable[[str], None] = print) -> dict[str, Any]:
    """Score a finished run's baseline and winners on another real-question file.

    Used for the unscoped version of the real questions. Nothing here feeds back into selection:
    the configs are read from best_configs.json as they were chosen.
    """
    best = json.loads((source_run / "best_configs.json").read_text())
    headline = best["headline_metric"]
    metrics = tuple(best.get("judged_metrics") or _judged_metrics(cfg, headline))
    configs = {"baseline": best["baseline"]["config"]}
    configs.update({arm_id: arm["best_config"] for arm_id, arm in best["arms"].items()})

    pairs = load_real_questions(cfg.dataset.real_questions_path)
    real_n = cfg.optimization.get("final_eval", {}).get("real_sample_size")
    log(f"{len(configs)} configs from {source_run.name} on {cfg.dataset.real_questions_path}: "
        f"retrieval on {len(pairs)}, judging {min(real_n or len(pairs), len(pairs))}")

    timing = TimingLog()
    factory = RagFactory(cfg, timing)
    final = _FinalEval(factory, build_judge(cfg.models.judge, cfg.runtime.cache_dir), metrics,
                       out_dir, log=log)
    scores = {}
    for label, config in configs.items():
        scores[label] = final(f"{label}__real_heldout", params_from_dict(config["retrieval"]),
                              config["answer_prompt"], pairs, real_n)

    report = {
        "source_run": str(source_run),
        "real_questions_path": cfg.dataset.real_questions_path,
        "headline_metric": headline,
        "judged_metrics": list(metrics),
        "configs": {label: {**configs[label], "scores": scores[label]} for label in configs},
        "significance": {"real_heldout": final.compare_real(list(best["arms"]), headline)},
    }
    write_json(out_dir / "reeval.json", report)
    timing.dump(out_dir / "timings.json")
    return report


def format_reeval(report: dict[str, Any]) -> str:
    headline = report["headline_metric"]
    rows = [("Config", headline, "faithfulness", "passage_hit", "paper_hit")]
    for label, block in report["configs"].items():
        judged, retrieval = block["scores"].get("judged", {}), block["scores"]["retrieval"]
        name = "Baseline (untuned)" if label == "baseline" else f"Optimized — {label}"
        rows.append((name, _fmt(judged.get(headline)), _fmt(judged.get("faithfulness")),
                     _fmt(retrieval.get("passage_hit_rate")), _fmt(retrieval.get("hit_rate"))))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def line(r):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(r))

    out = [f"Questions: {report['real_questions_path']}", "", line(rows[0]),
           "  ".join("-" * w for w in widths)]
    return "\n".join(out + [line(r) for r in rows[1:]])


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)


def format_table(report: dict[str, Any]) -> str:
    metric = report["headline_metric"]
    rows = [("Setup", "Synth val", "Synth test", "Real held-out")]
    b = report["baseline"]["scores"]
    rows.append(("Baseline (untuned)", "—", _fmt(b["synthetic_test_mean"]),
                 _fmt(b["real_heldout"])))
    for arm_id, arm in report["arms"].items():
        s = arm["scores"]
        rows.append((f"Optimized — {arm_id}", _fmt(s["synthetic_val"]),
                     _fmt(s["synthetic_test"]), _fmt(s["real_heldout"])))
    widths = [max(len(r[i]) for r in rows) for i in range(4)]

    def line(r):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(r))

    out = [f"Headline metric: {metric} (higher is better)", "", line(rows[0]),
           "  ".join("-" * w for w in widths)]
    out += [line(r) for r in rows[1:]]
    if not report.get("has_real_heldout", True):
        out += ["", "NOTE: no real questions provided, so these are synthetic-eval scores only. "
                    "Set dataset.real_questions_path to measure transfer."]
    return "\n".join(out)


def format_significance(report: dict[str, Any]) -> str:
    lines: list[str] = []
    for split, comparisons in (report.get("significance") or {}).items():
        for name, comp in comparisons.items():
            title = f"[{split}] {name.replace('_vs_', ' vs ')}"
            lines.append("")
            lines.extend(format_comparison(f"{title} · retrieval", comp["retrieval"]))
            if comp["judged"]["metrics"]:
                lines.extend(format_comparison(f"{title} · answers", comp["judged"]))
    if lines:
        lines += ["", SIG_NOTE]
    return "\n".join(lines).lstrip("\n")
