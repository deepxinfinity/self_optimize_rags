"""Re-score a finished run's configs on the real questions, retrieval only, with paired tests.

No LLM calls, so it takes minutes (seconds once embeddings are cached). Configs are replayed
through the current pipeline, including retrieval.context_budget_words, so runs written by an
older version are not reproduced bit for bit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from ..config import Config
from ..evalgen.generator import QAPair
from ..evalgen.heldout import load_real_questions
from ..evaluate import evaluate_retrieval
from ..params import RetrievalParams, baseline_retrieval_params, params_from_dict
from ..rag import RagFactory
from ..utils.runs import write_json, write_jsonl
from ..utils.timing import TimingLog
from .paired import RETRIEVAL_TESTS, SIG_NOTE, compare_records, format_comparison


def configs_from_run(run_dir: Path, cfg: Config) -> dict[str, RetrievalParams]:
    best_path = run_dir / "best_configs.json"
    if not best_path.exists():
        raise FileNotFoundError(f"{best_path} not found; is that an optimize run directory?")
    best = json.loads(best_path.read_text())

    baseline_block = (best.get("baseline") or {}).get("config", {}).get("retrieval")
    named = {"baseline": params_from_dict(baseline_block) if baseline_block
             else baseline_retrieval_params(cfg)}
    for arm_id, arm in (best.get("arms") or {}).items():
        retrieval = (arm.get("best_config") or {}).get("retrieval")
        if retrieval:
            named[arm_id] = params_from_dict(retrieval)
    if len(named) < 2:
        raise ValueError(f"{best_path} has no optimized arms to compare against the baseline")
    return named


def score_configs(cfg: Config, named: dict[str, RetrievalParams], pairs: list[QAPair],
                  timing: Optional[TimingLog] = None,
                  log: Optional[Callable[[str], None]] = None) -> dict[str, dict]:
    timing = timing or TimingLog()
    factory = RagFactory(cfg, timing)
    out: dict[str, dict] = {}
    for name in sorted(named, key=lambda n: (named[n].chunk_size, named[n].chunk_overlap)):
        params = named[name]
        if log:
            log(f"  · scoring '{name}' (cs={params.chunk_size} ov={params.chunk_overlap} "
                f"k={params.top_k} rr={'on' if params.use_reranker else 'off'}) "
                f"on {len(pairs)} real questions")
        metrics, records = evaluate_retrieval(factory.build(params), pairs, timing)
        out[name] = {"params": vars(params).copy(), "retrieval": metrics, "per_question": records}
    return out


def run_comparison(cfg: Config, run_dir: Path, out_dir: Path, config_path: str,
                   seed: int = 42, log: Optional[Callable[[str], None]] = None) -> dict:
    pairs = load_real_questions(cfg.dataset.real_questions_path)
    named = configs_from_run(run_dir, cfg)
    if log:
        log(f"{len(pairs)} real questions; configs: {', '.join(named)}")

    timing = TimingLog()
    scored = score_configs(cfg, named, pairs, timing, log=log)
    comparisons = {name: compare_records(block["per_question"],
                                         scored["baseline"]["per_question"],
                                         RETRIEVAL_TESTS, seed=seed)
                   for name, block in scored.items() if name != "baseline"}

    for name, block in scored.items():
        write_jsonl(out_dir / "per_question" / f"{name}.jsonl", block["per_question"])
    report = {
        "source_run": str(run_dir),
        "config": config_path,
        "real_questions_path": cfg.dataset.real_questions_path,
        "n_questions": len(pairs),
        "configs": {name: {"params": b["params"], "retrieval": b["retrieval"]}
                    for name, b in scored.items()},
        "paired_vs_baseline": comparisons,
        "seed": seed,
    }
    write_json(out_dir / "comparison.json", report)
    timing.dump(out_dir / "timings.json")
    return report


def format_report(report: dict) -> str:
    lines = [f"=== retrieval on real questions (n={report['n_questions']}) ===",
             f"{'config':<14}{'hit':>8}{'mrr':>8}{'p_hit':>8}{'p_mrr':>8}   params",
             "-" * 86]
    for name, block in report["configs"].items():
        r, p = block["retrieval"], block["params"]
        knobs = (f"cs={p['chunk_size']} ov={p['chunk_overlap']} k={p['top_k']} "
                 f"a={round(p['hybrid_alpha'], 3)} rr={'on' if p['use_reranker'] else 'off'}")
        passage = (f"{r['passage_hit_rate']:>8.4f}{r['passage_mrr']:>8.4f}"
                   if "passage_mrr" in r else f"{'—':>8}{'—':>8}")
        lines.append(f"{name:<14}{r['hit_rate']:>8.4f}{r['mrr']:>8.4f}{passage}   {knobs}")
    for name, comp in report["paired_vs_baseline"].items():
        lines.append("")
        lines.extend(format_comparison(f"{name} vs baseline", comp))
    lines += ["", SIG_NOTE]
    return "\n".join(lines)
