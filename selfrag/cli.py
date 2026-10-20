from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .config import load_config
from .evalgen.generator import build_eval_chunks, generate_arm_evalset
from .evalgen.splits import load_jsonl, split_pairs, write_jsonl
from .evaluate import evaluate
from .models.factory import build_judge
from .models.ollama_adapter import OllamaUnavailable, check_ollama
from .params import baseline_retrieval_params, params_from_dict
from .rag import RagFactory, build_rag
from .utils.runs import new_run_dir, write_json, write_run_meta
from .utils.runs import write_jsonl as write_rows
from .utils.timing import TimingLog


def _err(msg: str) -> None:
    print(f"» {msg}", file=sys.stderr)


def _common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--config", required=True, help="path to a YAML config")
    sub.add_argument("--set", dest="overrides", action="append", default=[], metavar="k.k=v",
                     help="override a config key (repeatable), e.g. --set search_space.top_k.low=5")


def _require_ollama(*specs) -> None:
    dicts = [s.model_dump() if hasattr(s, "model_dump") else dict(s) for s in specs if s]
    names = [d["name"] for d in dicts if d.get("provider") == "ollama"]
    if names:
        check_ollama(names)


def _answer_models(cfg) -> list:
    """Generator, judge and the stage-2 proposer (the judge's model unless set separately)."""
    return [cfg.models.generator, cfg.models.judge,
            cfg.optimization.get("stage2_prompts", {}).get("proposer")]


def _evalset_dir(cfg) -> Path:
    return Path(cfg.evalgen.get("out_dir", "data/evalsets"))


def cmd_ask(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overrides)
    _require_ollama(cfg.models.generator)
    params = baseline_retrieval_params(cfg)
    _err(f"building RAG (baseline params: {params})")
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
    _err(f"ingesting up to {args.n_papers} papers / {args.n_questions} questions "
         f"from {args.source}")
    summary = ingest_qasper(
        source=args.source, corpus_dir=args.corpus_dir, real_out=args.real_out,
        n_papers=args.n_papers, n_questions=args.n_questions, seed=args.seed,
        scope_with_title=args.scope_with_title, require_evidence=args.require_evidence,
        unscoped_out=args.unscoped_out,
    )
    print(f"wrote {summary['papers']} papers -> {args.corpus_dir}")
    print(f"wrote {summary['questions']} of {summary['answerable']} answerable questions "
          f"({summary['with_evidence']} with evidence, {summary['gold_papers_covered']} papers) "
          f"-> {args.real_out}")
    if args.unscoped_out:
        print(f"wrote the same questions without the title -> {args.unscoped_out}")
    return 0


def _generate_evalsets(cfg, arms) -> None:
    _require_ollama(*arms)
    timing = TimingLog()
    with timing.timed("build_eval_chunks") as rec:
        chunks = build_eval_chunks(cfg)
        rec["n_chunks"] = len(chunks)
    _err(f"{len(chunks)} source passages; generating for {len(arms)} arm(s)")

    for arm in arms:
        _err(f"{arm.id} ({arm.name}): generating {cfg.evalgen.get('n_pairs_per_arm', 100)} pairs, "
             f"the first call also loads the model")
        with timing.timed("generate_arm", arm=arm.id) as rec:
            pairs, stats = generate_arm_evalset(cfg, arm, chunks, timing, log=_err)
            rec["n_pairs"] = len(pairs)
        splits = split_pairs(pairs, cfg.splits.train, cfg.splits.val, cfg.splits.test,
                             seed=cfg.splits.seed)

        arm_dir = _evalset_dir(cfg) / arm.id
        write_jsonl(pairs, arm_dir / "all.jsonl")
        for name, subset in splits.items():
            write_jsonl(subset, arm_dir / f"{name}.jsonl")
        write_json(arm_dir / "manifest.json", {
            "arm_id": arm.id,
            "model": arm.name,
            "n_pairs": len(pairs),
            "splits": {k: len(v) for k, v in splits.items()},
            "generation": stats,
            "seed": cfg.splits.seed,
            "corpus_path": str(Path(cfg.dataset.corpus_path).resolve()),
        })
        print(f"  arm {arm.id:<10} ({arm.name}): {len(pairs)} pairs -> "
              f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
              f"  rejected: {stats.get('unparsed', 0)} unparsed, {stats.get('aborted', 0)} aborted, "
              f"{stats.get('ungrounded', 0)} ungrounded, {stats.get('duplicate', 0)} duplicate")

    print("\n=== TIMINGS ===")
    print(timing.summary())
    _err(f"spot-check ~{cfg.evalgen.get('manual_spotcheck', 50)} pairs per arm before optimizing")


def cmd_gen_eval(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overrides)
    _generate_evalsets(cfg, cfg.models.eval_generators)
    return 0


def _evalset_matches(cfg, arm) -> bool:
    manifest = _evalset_dir(cfg) / arm.id / "manifest.json"
    if not manifest.exists():
        return False
    made_from = json.loads(manifest.read_text()).get("corpus_path")
    # sets written before the corpus was recorded are trusted as they are
    return made_from is None or Path(made_from) == Path(cfg.dataset.corpus_path).resolve()


def cmd_pipeline(args: argparse.Namespace) -> int:
    """gen-eval for arms without an eval set for this corpus (or all with --force), then optimize.

    The web job runner spawns this so one pid covers the whole job.
    """
    cfg = load_config(args.config, args.overrides)
    need = [a for a in cfg.models.eval_generators
            if args.force or not _evalset_matches(cfg, a)]
    _require_ollama(*_answer_models(cfg))
    if need:
        _err(f"pipeline: generating eval sets for {[a.id for a in need]}")
        _generate_evalsets(cfg, need)
    else:
        _err("pipeline: eval sets exist for all arms (use --force to regenerate)")
    return cmd_optimize(args)


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overrides)
    split_path = _evalset_dir(cfg) / args.arm / f"{args.split}.jsonl"
    if not split_path.exists():
        _err(f"error: {split_path} not found, run gen-eval first")
        return 1
    if args.split == "test":
        _err("WARNING: the test split is meant to be scored once, by the optimize run")

    if args.judged:
        _require_ollama(cfg.models.generator, cfg.models.judge)
    pairs = load_jsonl(split_path)
    timing = TimingLog()
    _err(f"scoring the baseline on {len(pairs)} '{args.split}' pairs for arm '{args.arm}'")
    rag = build_rag(cfg, baseline_retrieval_params(cfg), timing)
    judge = build_judge(cfg.models.judge, cfg.runtime.cache_dir) if args.judged else None
    sample_size = args.sample or cfg.optimization.get("stage2_prompts", {}).get("sample_size")

    with timing.timed("evaluate"):
        report = evaluate(rag, pairs, judge=judge, sample_size=sample_size, timing=timing)

    run_dir = new_run_dir(cfg.runtime.out_dir, tag=f"eval-{args.arm}-{args.split}")
    write_run_meta(run_dir, cfg.dataset.corpus_path, args.config, "eval", arm=args.arm,
                   split=args.split, overrides=args.overrides)
    for kind, rows in report.pop("per_question").items():
        write_rows(run_dir / "per_question" / f"{kind}.jsonl", rows)
    report.update(arm=args.arm, split=args.split, baseline=True)
    write_json(run_dir / "eval.json", report)
    timing.dump(run_dir / "timings.json")

    print("\n=== EVAL REPORT ===")
    print(f"arm={args.arm} split={args.split} n={report['n_pairs']}")
    print(f"retrieval: {report['retrieval']}")
    if "judged" in report:
        print(f"judged: {report['judged']}")
    print(f"\nwrote {run_dir}")
    return 0


def cmd_stage1(args: argparse.Namespace) -> int:
    from .optimize.stage1_retrieval import Stage1Optimizer
    cfg = load_config(args.config, args.overrides)
    val_path = _evalset_dir(cfg) / args.arm / "val.jsonl"
    if not val_path.exists():
        _err(f"error: {val_path} not found, run gen-eval first")
        return 1
    pairs = load_jsonl(val_path)

    run_dir = new_run_dir(cfg.runtime.out_dir, tag=f"stage1-{args.arm}")
    write_run_meta(run_dir, cfg.dataset.corpus_path, args.config, "stage1", arm=args.arm,
                   overrides=args.overrides)
    opt = Stage1Optimizer(cfg, pairs, trial_log_path=run_dir / "stage1_trials.jsonl")
    _err(f"stage 1 on '{args.arm}' val ({len(pairs)} pairs), metric={opt.metric}, "
         f"n_trials={opt.n_trials}")
    result = opt.run()

    write_json(run_dir / "stage1_best.json", {
        "arm": args.arm,
        "stage": "stage1_retrieval",
        "metric": result.metric,
        "baseline_value": result.baseline_value,
        "best_value": result.best_value,
        "best_params": asdict(result.best_params),
        "best_metrics": result.best_metrics,
        "n_trials": result.n_trials,
        "total_seconds": result.total_seconds,
    })
    print(f"\n{result.metric}: baseline {result.baseline_value} -> best {result.best_value}"
          f"  ({result.n_trials} trials, {result.total_seconds:.0f}s)")
    print(f"best params: {result.best_params}")
    print(f"wrote {run_dir}")
    return 0


def cmd_stage2(args: argparse.Namespace) -> int:
    from .optimize.stage2_prompts import Stage2Optimizer
    cfg = load_config(args.config, args.overrides)
    arm_dir = _evalset_dir(cfg) / args.arm
    if not (arm_dir / "val.jsonl").exists():
        _err(f"error: {arm_dir / 'val.jsonl'} not found, run gen-eval first")
        return 1
    _require_ollama(*_answer_models(cfg))
    val = load_jsonl(arm_dir / "val.jsonl")
    train = load_jsonl(arm_dir / "train.jsonl") if (arm_dir / "train.jsonl").exists() else []

    if args.stage1:
        frozen = params_from_dict(json.loads(Path(args.stage1).read_text())["best_params"])
        _err(f"retrieval frozen from {args.stage1}")
    else:
        frozen = baseline_retrieval_params(cfg)
        _err("no --stage1 given, freezing the baseline retrieval config")

    run_dir = new_run_dir(cfg.runtime.out_dir, tag=f"stage2-{args.arm}")
    write_run_meta(run_dir, cfg.dataset.corpus_path, args.config, "stage2", arm=args.arm,
                   overrides=args.overrides)
    opt = Stage2Optimizer(cfg, val, frozen, train_pairs=train,
                          trial_log_path=run_dir / "stage2_trials.jsonl",
                          factory=RagFactory(cfg))
    _err(f"stage 2 on '{args.arm}', metric={opt.metric}, n_trials={opt.n_trials}")
    result = opt.run()

    write_json(run_dir / "stage2_best.json", {
        "arm": args.arm,
        "stage": "stage2_prompts",
        "metric": result.metric,
        "seed_value": result.seed_value,
        "best_value": result.best_value,
        "best_prompt": result.best_prompt,
        "frozen_retrieval": asdict(frozen),
        "n_trials": result.n_trials,
        "total_seconds": result.total_seconds,
    })
    print(f"\n{result.metric}: seed {result.seed_value} -> best {result.best_value}"
          f"  ({result.n_trials} trials, {result.total_seconds:.0f}s)")
    print(f"best prompt:\n{result.best_prompt}")
    print(f"wrote {run_dir}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    """Embed every chunking in the search space into the embedding cache ahead of a run."""
    cfg = load_config(args.config, args.overrides)
    if not cfg.runtime.embedding_cache_dir:
        _err("error: runtime.embedding_cache_dir is not set, nothing would be kept")
        return 1
    ss = cfg.search_space
    base = baseline_retrieval_params(cfg)
    grid = {(base.chunk_size, base.chunk_overlap)}
    grid |= {(size, overlap) for size in ss["chunk_size"]["values"]
             for overlap in ss["chunk_overlap"]["values"] if overlap < size}
    factory = RagFactory(cfg)
    _err(f"embedding {len(grid)} chunkings with {cfg.models.embedding.name} "
         f"on {cfg.models.embedding.model_dump().get('device') or 'the default device'}")
    for size, overlap in sorted(grid):
        with factory.timing.timed("chunking", chunk_size=size, overlap=overlap):
            chunks, _ = factory.index_for(size, overlap)
        _err(f"  {size}/{overlap}: {len(chunks)} chunks")
    print(factory.timing.summary())
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    import uvicorn
    from .ui.app import create_app
    app = create_app(args.config)
    _err(f"UI on http://{args.host}:{args.port}  (config: {args.config})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    from .optimize.runner import format_significance, format_table, run_optimize
    cfg = load_config(args.config, args.overrides)
    _require_ollama(*_answer_models(cfg))
    run_dir = new_run_dir(cfg.runtime.out_dir, tag="optimize")
    write_run_meta(run_dir, cfg.dataset.corpus_path, args.config, "optimize",
                   overrides=args.overrides)
    _err(f"optimizing {len(cfg.models.eval_generators)} arm(s); run dir: {run_dir}")
    report = run_optimize(cfg, run_dir, log=_err)

    print("\n=== RESULTS ===")
    print(format_table(report))
    sig = format_significance(report)
    if sig:
        print("\n=== PAIRED TESTS ===")
        print(sig)
    print(f"\nwrote {run_dir}/best_configs.json")
    print("\nheld-out access log:")
    for name, accesses in report["heldout_access_log"].items():
        print(f"  {name}: {accesses}")
    return 0


def cmd_reeval(args: argparse.Namespace) -> int:
    from .optimize.runner import format_reeval, format_significance, run_reeval
    cfg = load_config(args.config, args.overrides)
    run_dir = Path(args.run)
    if not (run_dir / "best_configs.json").exists():
        _err(f"error: {run_dir / 'best_configs.json'} not found")
        return 1
    _require_ollama(cfg.models.generator, cfg.models.judge)
    out_dir = new_run_dir(cfg.runtime.out_dir, tag="reeval")
    write_run_meta(out_dir, cfg.dataset.corpus_path, args.config, "reeval",
                   overrides=args.overrides)
    report = run_reeval(cfg, run_dir, out_dir, log=_err)
    print("\n=== RESULTS ===")
    print(format_reeval(report))
    print("\n=== PAIRED TESTS ===")
    print(format_significance(report))
    print(f"\nwrote {out_dir}")
    return 0


def cmd_compare_retrieval(args: argparse.Namespace) -> int:
    from .analysis.compare import format_report, run_comparison
    cfg = load_config(args.config, args.overrides)
    run_dir = Path(args.run)
    if not run_dir.exists():
        _err(f"error: run directory {run_dir} not found")
        return 1
    out_dir = new_run_dir(cfg.runtime.out_dir, tag="compare-retrieval")
    write_run_meta(out_dir, cfg.dataset.corpus_path, args.config, "compare",
                   overrides=args.overrides)
    _err(f"re-scoring configs from {run_dir} on the real questions (retrieval only)")
    report = run_comparison(cfg, run_dir, out_dir, args.config, seed=args.seed, log=_err)
    print()
    print(format_report(report))
    print(f"\nwrote {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="selfrag", description="Self-optimizing RAG")
    subs = parser.add_subparsers(dest="command", required=True)

    p = subs.add_parser("ask", help="answer one question with the baseline RAG")
    p.add_argument("question")
    _common(p)
    p.set_defaults(func=cmd_ask)

    p = subs.add_parser("ingest-qasper", help="build the QASPER corpus + real questions")
    p.add_argument("--source", default="data/qasper_raw/qasper-dev-v0.3.json")
    p.add_argument("--corpus-dir", dest="corpus_dir", default="data/corpus/qasper")
    p.add_argument("--real-out", dest="real_out", default="data/real_questions_qasper.jsonl")
    p.add_argument("--n-papers", dest="n_papers", type=int, default=50)
    p.add_argument("--n-questions", dest="n_questions", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scope-with-title", dest="scope_with_title", action="store_true",
                   help='rewrite questions as: In the paper "<title>": <question>')
    p.add_argument("--require-evidence", dest="require_evidence", action="store_true",
                   help="keep only questions with text evidence (needed for passage metrics)")
    p.add_argument("--unscoped-out", dest="unscoped_out", default=None,
                   help="with --scope-with-title, also write the same questions without the title")
    p.set_defaults(func=cmd_ingest_qasper)

    p = subs.add_parser("embed", help="pre-compute embeddings for every chunking in the search space")
    _common(p)
    p.set_defaults(func=cmd_embed)

    p = subs.add_parser("gen-eval", help="generate synthetic eval sets + splits for every arm")
    _common(p)
    p.set_defaults(func=cmd_gen_eval)

    p = subs.add_parser("eval", help="score the baseline config on one arm's split")
    p.add_argument("--arm", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--judged", action="store_true", help="also score answers with the judge")
    p.add_argument("--sample", type=int, default=None, help="cap the number of judged pairs")
    _common(p)
    p.set_defaults(func=cmd_eval)

    p = subs.add_parser("stage1", help="tune retrieval on one arm's val split")
    p.add_argument("--arm", required=True)
    _common(p)
    p.set_defaults(func=cmd_stage1)

    p = subs.add_parser("stage2", help="tune the answer prompt on one arm's val split")
    p.add_argument("--arm", required=True)
    p.add_argument("--stage1", default=None, help="stage1_best.json to freeze retrieval from")
    _common(p)
    p.set_defaults(func=cmd_stage2)

    p = subs.add_parser("ui", help="dashboard + chat + projects")
    p.add_argument("--config", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_ui)

    p = subs.add_parser("optimize", help="both stages for every arm, then the held-out report")
    _common(p)
    p.set_defaults(func=cmd_optimize)

    p = subs.add_parser("compare-retrieval",
                        help="re-score a finished run on the real questions with paired tests")
    p.add_argument("--run", required=True, help="optimize run dir containing best_configs.json")
    p.add_argument("--seed", type=int, default=42, help="bootstrap seed")
    _common(p)
    p.set_defaults(func=cmd_compare_retrieval)

    p = subs.add_parser("reeval", help="score a finished run's configs on another real-question "
                                       "file (retrieval, judged answers, paired tests)")
    p.add_argument("--run", required=True, help="optimize run dir containing best_configs.json")
    _common(p)
    p.set_defaults(func=cmd_reeval)

    p = subs.add_parser("pipeline", help="gen-eval (if needed) then optimize, in one process")
    p.add_argument("--force", action="store_true", help="regenerate eval sets even if present")
    _common(p)
    p.set_defaults(func=cmd_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except OllamaUnavailable as e:
        _err(f"error: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
