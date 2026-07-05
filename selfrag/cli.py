"""Command-line surface: ``python -m selfrag <command> --config ... --set k=v``.

Commands:
  ask       Run the naive-baseline RAG on a single question (Increment 1).
  optimize  The product surface — two-arm, two-stage optimization (wired in later increments).

``--set a.b.c=value`` overrides any config key without editing the file.
"""

from __future__ import annotations

import argparse
import sys

import json
from pathlib import Path

from .config import load_config
from .evalgen.generator import build_eval_chunks, generate_arm_evalset
from .evalgen.splits import load_jsonl, split_pairs, write_jsonl
from .evaluate import evaluate
from .models.factory import build_judge
from .optimize.search_space import params_from_dict
from .optimize.stage1_retrieval import Stage1Optimizer
from .optimize.stage2_prompts import Stage2Optimizer
from .params import baseline_retrieval_params
from .rag import build_rag
from .utils.runs import new_run_dir, write_json, write_run_meta
from .utils.timing import TimingLog


def _common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--config", required=True, help="path to a YAML config")
    sub.add_argument("--set", dest="overrides", action="append", default=[], metavar="k.k=v",
                     help="override a config key (repeatable), e.g. --set search_space.top_k.low=5")


def cmd_ask(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overrides)
    params = baseline_retrieval_params(cfg)
    print(f"» building RAG (baseline params: {params})", file=sys.stderr)
    rag = build_rag(cfg, params)
    result = rag.ask(args.question)

    print("\n=== ANSWER ===")
    print(result.answer)
    print("\n=== SOURCES ===")
    for i, r in enumerate(result.sources, 1):
        print(f"[{i}] {r.chunk.chunk_id}  (score={r.score:.4f})")
    print("\n=== TIMINGS ===")
    print(rag.timing.summary())
    return 0


def cmd_ingest_qasper(args: argparse.Namespace) -> int:
    from .corpus.qasper import ingest_qasper
    print(f"» ingesting up to {args.n_papers} papers / {args.n_questions} real questions "
          f"from {args.source}", file=sys.stderr)
    summary = ingest_qasper(
        source=args.source, corpus_dir=args.corpus_dir, real_out=args.real_out,
        n_papers=args.n_papers, n_questions=args.n_questions, seed=args.seed,
    )
    print(f"wrote {summary['papers']} papers -> {args.corpus_dir}")
    print(f"wrote {summary['questions']} real questions "
          f"(covering {summary['gold_papers_covered']} papers) -> {args.real_out}")
    print("\nnext: selfrag gen-eval --config configs/qasper.yaml")
    return 0


def cmd_gen_eval(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overrides)
    timing = TimingLog()
    out_root = Path(cfg.evalgen.get("out_dir", "data/evalsets"))

    with timing.timed("build_eval_chunks") as rec:
        chunks = build_eval_chunks(cfg)
        rec["n_chunks"] = len(chunks)
    print(f"» {len(chunks)} source passages; generating for "
          f"{len(cfg.models.eval_generators)} arm(s)", file=sys.stderr)

    for arm in cfg.models.eval_generators:
        with timing.timed("generate_arm", arm=arm.id) as rec:
            pairs = generate_arm_evalset(cfg, arm, chunks, timing)
            rec["n_pairs"] = len(pairs)
        splits = split_pairs(pairs, cfg.splits.train, cfg.splits.val, cfg.splits.test,
                             seed=cfg.splits.seed)

        arm_dir = out_root / arm.id
        write_jsonl(pairs, arm_dir / "all.jsonl")
        for name, subset in splits.items():
            write_jsonl(subset, arm_dir / f"{name}.jsonl")
        manifest = {
            "arm_id": arm.id,
            "model": arm.name,
            "n_pairs": len(pairs),
            "splits": {k: len(v) for k, v in splits.items()},
            "seed": cfg.splits.seed,
        }
        (arm_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"  arm {arm.id:<10} ({arm.name}): {len(pairs)} pairs -> "
              f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
              f"  [{arm_dir}]")

    print("\n=== TIMINGS ===")
    print(timing.summary())
    print(f"\nReminder: hand-spot-check ~{cfg.evalgen.get('manual_spotcheck', 50)} pairs per arm "
          "for quality (logged, not enforced).", file=sys.stderr)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overrides)
    out_root = Path(cfg.evalgen.get("out_dir", "data/evalsets"))
    split_path = out_root / args.arm / f"{args.split}.jsonl"
    if not split_path.exists():
        print(f"error: {split_path} not found — run `selfrag gen-eval` first.", file=sys.stderr)
        return 1
    if args.split in ("test",):
        print("WARNING: scoring the synthetic 'test' split — per the study's one-touch rule this "
              "should happen once, at the very end (see the runner in Increment 6).", file=sys.stderr)

    pairs = load_jsonl(split_path)
    timing = TimingLog()
    params = baseline_retrieval_params(cfg)
    print(f"» building RAG (baseline) and scoring {len(pairs)} '{args.split}' pairs "
          f"for arm '{args.arm}'", file=sys.stderr)
    rag = build_rag(cfg, params, timing)

    judge = None
    if args.judged:
        judge = build_judge(cfg.models.judge, cfg.runtime.cache_dir)
    sample_size = args.sample or int(cfg.optimization.get("stage2_prompts", {}).get("sample_size", 0)) or None

    with timing.timed("evaluate"):
        report = evaluate(rag, pairs, judge=judge, sample_size=sample_size, timing=timing)

    run_dir = new_run_dir(cfg.runtime.out_dir, tag=f"eval-{args.arm}-{args.split}")
    write_run_meta(run_dir, cfg.dataset.corpus_path, args.config, "eval", arm=args.arm, split=args.split)
    report["arm"] = args.arm
    report["split"] = args.split
    report["baseline"] = True
    write_json(run_dir / "eval.json", report)
    timing.dump(run_dir / "timings.json")

    print("\n=== EVAL REPORT ===")
    print(f"arm={args.arm} split={args.split} n={report['n_pairs']}")
    print(f"retrieval: hit_rate={report['retrieval']['hit_rate']}  mrr={report['retrieval']['mrr']}")
    if "judged" in report:
        j = report["judged"]
        print(f"judged (n={j['n']}): faithfulness={j['faithfulness']}  "
              f"answer_relevancy={j['answer_relevancy']}")
    print(f"\nwrote {run_dir}/eval.json + timings.json")
    print("\n=== TIMINGS ===")
    print(timing.summary())
    return 0


def cmd_stage1(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overrides)
    out_root = Path(cfg.evalgen.get("out_dir", "data/evalsets"))
    # Stage 1 SELECTS against val — never test/real.
    val_path = out_root / args.arm / "val.jsonl"
    if not val_path.exists():
        print(f"error: {val_path} not found — run `selfrag gen-eval` first.", file=sys.stderr)
        return 1
    pairs = load_jsonl(val_path)

    run_dir = new_run_dir(cfg.runtime.out_dir, tag=f"stage1-{args.arm}")
    write_run_meta(run_dir, cfg.dataset.corpus_path, args.config, "stage1", arm=args.arm)
    trial_log = run_dir / "stage1_trials.jsonl"
    opt = Stage1Optimizer(cfg, pairs, trial_log_path=trial_log)
    print(f"» Stage 1: optimizing retrieval on '{args.arm}' val ({len(pairs)} pairs), "
          f"metric={opt.metric}, n_trials={opt.n_trials}", file=sys.stderr)
    result = opt.run()

    from dataclasses import asdict
    payload = {
        "arm": args.arm,
        "stage": "stage1_retrieval",
        "metric": result.metric,
        "baseline_value": result.baseline_value,
        "best_value": result.best_value,
        "best_params": asdict(result.best_params),
        "n_trials": result.n_trials,
        "total_seconds": result.total_seconds,
    }
    write_json(run_dir / "stage1_best.json", payload)

    delta = round(result.best_value - result.baseline_value, 4)
    print("\n=== STAGE 1 RESULT ===")
    print(f"arm={args.arm}  metric={result.metric}  trials={result.n_trials}  "
          f"time={result.total_seconds}s")
    print(f"baseline {result.metric} = {result.baseline_value}")
    print(f"best     {result.metric} = {result.best_value}   (delta {delta:+})")
    print(f"best params: {result.best_params}")
    print(f"\nwrote {run_dir}/stage1_best.json + stage1_trials.jsonl")
    return 0


def cmd_stage2(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overrides)
    out_root = Path(cfg.evalgen.get("out_dir", "data/evalsets"))
    val_path = out_root / args.arm / "val.jsonl"
    if not val_path.exists():
        print(f"error: {val_path} not found — run `selfrag gen-eval` first.", file=sys.stderr)
        return 1
    pairs = load_jsonl(val_path)

    # Frozen retrieval: from a Stage-1 result if given, else the naive baseline.
    if args.stage1:
        best = json.loads(Path(args.stage1).read_text())["best_params"]
        frozen = params_from_dict(best)
        print(f"» frozen retrieval from {args.stage1}", file=sys.stderr)
    else:
        frozen = baseline_retrieval_params(cfg)
        print("» no --stage1 given; freezing the naive baseline retrieval", file=sys.stderr)

    run_dir = new_run_dir(cfg.runtime.out_dir, tag=f"stage2-{args.arm}")
    write_run_meta(run_dir, cfg.dataset.corpus_path, args.config, "stage2", arm=args.arm)
    trial_log = run_dir / "stage2_trials.jsonl"
    opt = Stage2Optimizer(cfg, pairs, frozen, trial_log_path=trial_log)
    print(f"» Stage 2: optimizing answer prompt on '{args.arm}' val, method={opt.method}, "
          f"metric={opt.metric}, n_trials={opt.n_trials}", file=sys.stderr)
    result = opt.run()

    from dataclasses import asdict
    payload = {
        "arm": args.arm,
        "stage": "stage2_prompts",
        "metric": result.metric,
        "seed_value": result.seed_value,
        "best_value": result.best_value,
        "best_prompt": result.best_prompt,
        "frozen_retrieval": asdict(frozen),
        "n_trials": result.n_trials,
        "total_seconds": result.total_seconds,
    }
    write_json(run_dir / "stage2_best.json", payload)

    delta = round(result.best_value - result.seed_value, 4)
    print("\n=== STAGE 2 RESULT ===")
    print(f"arm={args.arm}  metric={result.metric}  trials={result.n_trials}  "
          f"time={result.total_seconds}s")
    print(f"seed {result.metric} = {result.seed_value}")
    print(f"best {result.metric} = {result.best_value}   (delta {delta:+})")
    print(f"\nbest prompt:\n{result.best_prompt}")
    print(f"\nwrote {run_dir}/stage2_best.json + stage2_trials.jsonl")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    import uvicorn
    from .ui.app import create_app
    app = create_app(args.config)
    print(f"» Self-Optimizing RAG UI on http://{args.host}:{args.port}  (config: {args.config})",
          file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overrides)
    from .optimize.runner import format_table, run_optimize
    run_dir = new_run_dir(cfg.runtime.out_dir, tag="optimize")
    write_run_meta(run_dir, cfg.dataset.corpus_path, args.config, "optimize")
    print(f"» self-optimizing {len(cfg.models.eval_generators)} arm(s) through both stages; "
          f"run dir: {run_dir}", file=sys.stderr)
    report = run_optimize(cfg, run_dir)

    print("\n=== RESULTS ===")
    print(format_table(report))
    print(f"\nwrote {run_dir}/best_configs.json (+ per-arm stageN_trials.jsonl)")
    print("\nheld-out access log (each should be touched only in the final phase):")
    for name, accesses in report["heldout_access_log"].items():
        print(f"  {name}: {accesses}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="selfrag", description="Self-Optimizing RAG")
    subs = parser.add_subparsers(dest="command", required=True)

    p_ask = subs.add_parser("ask", help="answer a single question with the baseline RAG")
    p_ask.add_argument("question", help="the question to answer")
    _common(p_ask)
    p_ask.set_defaults(func=cmd_ask)

    p_ing = subs.add_parser("ingest-qasper", help="build a QASPER corpus + real questions from the JSON")
    p_ing.add_argument("--source", default="data/qasper_raw/qasper-dev-v0.3.json")
    p_ing.add_argument("--corpus-dir", dest="corpus_dir", default="data/corpus/qasper")
    p_ing.add_argument("--real-out", dest="real_out", default="data/real_questions_qasper.jsonl")
    p_ing.add_argument("--n-papers", dest="n_papers", type=int, default=50)
    p_ing.add_argument("--n-questions", dest="n_questions", type=int, default=100)
    p_ing.add_argument("--seed", type=int, default=42)
    p_ing.set_defaults(func=cmd_ingest_qasper)

    p_gen = subs.add_parser("gen-eval", help="generate synthetic eval sets per arm + splits")
    _common(p_gen)
    p_gen.set_defaults(func=cmd_gen_eval)

    p_eval = subs.add_parser("eval", help="score the baseline config on an arm's split")
    p_eval.add_argument("--arm", required=True, help="eval-set arm id (e.g. local_qg)")
    p_eval.add_argument("--split", default="val", choices=["train", "val", "test"])
    p_eval.add_argument("--judged", action="store_true", help="also run judged metrics (LLM judge)")
    p_eval.add_argument("--sample", type=int, default=None, help="cap pairs for judged metrics")
    _common(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    p_s1 = subs.add_parser("stage1", help="Stage 1: optimize retrieval knobs on an arm's val split")
    p_s1.add_argument("--arm", required=True, help="eval-set arm id (e.g. local_qg)")
    _common(p_s1)
    p_s1.set_defaults(func=cmd_stage1)

    p_s2 = subs.add_parser("stage2", help="Stage 2: optimize the answer prompt on an arm's val split")
    p_s2.add_argument("--arm", required=True, help="eval-set arm id (e.g. local_qg)")
    p_s2.add_argument("--stage1", default=None, help="path to a stage1_best.json to freeze retrieval")
    _common(p_s2)
    p_s2.set_defaults(func=cmd_stage2)

    p_ui = subs.add_parser("ui", help="launch the local web dashboard + chat")
    p_ui.add_argument("--config", required=True, help="path to a YAML config")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8000)
    p_ui.set_defaults(func=cmd_ui)

    p_opt = subs.add_parser("optimize", help="run the full self-optimizer: both arms, both stages, held-out eval")
    _common(p_opt)
    p_opt.set_defaults(func=cmd_optimize)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
