"""Local dashboard over runs/, chat against the baseline or latest optimized config, and the
projects API from selfrag/web."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..config import load_config
from ..params import baseline_retrieval_params, params_from_dict
from ..rag import RagFactory
from ..utils.runs import optimize_progress as _optimize_progress
from ..utils.runs import pid_alive as _pid_alive
from ..utils.timing import TimingLog

_HERE = Path(__file__).parent


class ChatReq(BaseModel):
    question: str
    source: str = "baseline"   # or "best"
    arm: Optional[str] = None


def _read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _classify(name: str) -> tuple[str, str, str]:
    """'<YYYYMMDD>-<HHMMSS>-<tag>' -> (kind, timestamp, tag)."""
    parts = name.split("-", 2)
    if len(parts) < 3:
        return "other", name, name
    ts, tag = f"{parts[0]}-{parts[1]}", parts[2]
    return tag.split("-")[0], ts, tag


def _parse_tag(kind: str, tag: str) -> tuple[Optional[str], Optional[str]]:
    """tag 'stage1-frontier' -> arm=frontier; 'eval-local_qg-val' -> arm, split."""
    rest = tag.split("-")[1:]
    if kind == "eval":
        return (rest[0] if rest else None), (rest[1] if len(rest) > 1 else None)
    return (rest[0] if rest else None), None


def _describe(kind: str, arm: Optional[str], split: Optional[str]) -> tuple[str, str]:
    a = f" · arm: {arm}" if arm else ""
    if kind == "optimize":
        return ("Full optimization",
                "Runs both eval-generator arms through Stage 1 (retrieval) and Stage 2 (prompts), "
                "then measures the winners on held-out synthetic test + the real questions.")
    if kind == "stage1":
        return (f"Stage 1: retrieval tuning{a}",
                f"Searches chunk size/overlap, top-k, dense/keyword mix and reranker to maximise "
                f"retrieval quality (MRR) on the '{arm}' synthetic questions.")
    if kind == "stage2":
        return (f"Stage 2: prompt tuning{a}",
                f"Retrieval frozen; rewrites the answer prompt to maximise judged answer quality "
                f"answer quality on the '{arm}' synthetic questions.")
    if kind == "eval":
        return (f"Evaluation{a}" + (f" · {split}" if split else ""),
                "Scores one configuration's retrieval + judged metrics on a single split.")
    if kind == "reeval":
        return ("Re-evaluation",
                "Scores a finished run's baseline and winners on another set of real questions.")
    if kind == "compare":
        return ("Retrieval comparison",
                "Re-scores a finished run's configs on the real questions, retrieval only.")
    return (kind, "")


_DONE_FILE = {"optimize": "best_configs.json", "stage1": "stage1_best.json",
              "stage2": "stage2_best.json", "eval": "eval.json", "reeval": "reeval.json",
              "compare": "comparison.json"}

# runs from before run_meta.json recorded a pid count as running if written to recently
_ACTIVE_WINDOW_SEC = 180


def _status(d: Path, kind: str) -> str:
    done = _DONE_FILE.get(kind)
    if done and (d / done).exists():
        return "complete"
    meta = _read_json(d / "run_meta.json") or {}
    if _pid_alive(meta.get("pid")):
        return "running"
    if meta.get("pid"):
        return "incomplete"
    newest = max((p.stat().st_mtime for p in d.iterdir()), default=0.0)
    return "running" if (time.time() - newest) < _ACTIVE_WINDOW_SEC else "incomplete"


def create_app(config_path: str) -> FastAPI:
    app = FastAPI(title="Self-Optimizing RAG")
    cfg = load_config(config_path)
    runs_dir = Path(cfg.runtime.out_dir)
    factory = RagFactory(cfg)
    rag_cache: dict[str, Any] = {}

    from ..web.routes import build_router
    app.include_router(build_router(cfg, config_path))

    def _run_summary(name: str, kind: str) -> dict:
        d = runs_dir / name
        if kind == "optimize":
            best = _read_json(d / "best_configs.json")
            if best:
                return {"headline_metric": best.get("headline_metric"),
                        "arms": {a: v["scores"] for a, v in best.get("arms", {}).items()},
                        "baseline": best.get("baseline", {}).get("scores"),
                        "complete": True}
            prog = _optimize_progress(d)
            trials = sum(sum(v.values()) for v in prog.values())
            return {"complete": False, "progress": prog, "trials_so_far": trials}
        if kind == "stage1":
            return _read_json(d / "stage1_best.json") or {"complete": False}
        if kind == "stage2":
            return _read_json(d / "stage2_best.json") or {"complete": False}
        if kind == "eval":
            return _read_json(d / "eval.json") or {}
        return {}

    @app.get("/api/summary")
    def summary():
        corpus = Path(cfg.dataset.corpus_path)
        n_docs = len(list(corpus.rglob("*.txt"))) + len(list(corpus.rglob("*.md"))) if corpus.exists() else 0
        return {
            "config": config_path,
            "corpus_path": str(corpus),
            "n_docs": n_docs,
            "arms": [a.id for a in cfg.models.eval_generators],
            "generator": cfg.models.generator.name,
            "judge": cfg.models.judge.name,
        }

    @app.get("/api/runs")
    def list_runs():
        if not runs_dir.exists():
            return {"current_dataset": cfg.dataset.corpus_path, "runs": []}
        out = []
        for d in sorted(runs_dir.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            kind, ts, tag = _classify(d.name)
            meta = _read_json(d / "run_meta.json") or {}
            arm = meta.get("arm") or _parse_tag(kind, tag)[0]
            split = meta.get("split") or _parse_tag(kind, tag)[1]
            label, description = _describe(kind, arm, split)
            status = _status(d, kind)
            out.append({
                "name": d.name, "kind": kind, "timestamp": ts, "tag": tag,
                "label": label, "description": description,
                "arm": arm, "split": split, "status": status,
                "dataset": meta.get("corpus_path"),
                "summary": _run_summary(d.name, kind),
            })
        return {"current_dataset": cfg.dataset.corpus_path, "runs": out}

    @app.get("/api/runs/{name}")
    def run_detail(name: str):
        d = runs_dir / name
        if not d.is_dir():
            return JSONResponse({"error": "not found"}, status_code=404)
        kind, ts, tag = _classify(name)
        trials: dict[str, list[dict]] = {}
        for f in sorted(d.glob("*_trials.jsonl")):
            trials[f.stem] = _read_jsonl(f)
        for f in sorted(d.glob("stage*_trials.jsonl")):
            trials.setdefault(f.stem, _read_jsonl(f))
        return {
            "name": name, "kind": kind, "tag": tag, "timestamp": ts,
            "best_configs": _read_json(d / "best_configs.json"),
            "stage1_best": _read_json(d / "stage1_best.json"),
            "stage2_best": _read_json(d / "stage2_best.json"),
            "eval": _read_json(d / "eval.json"),
            "timings": _read_json(d / "timings.json"),
            "trials": trials,
        }

    def _get_rag(source: str, arm: Optional[str]):
        key = f"{source}:{arm or ''}"
        if key in rag_cache:
            return rag_cache[key]
        params = baseline_retrieval_params(cfg)
        answer_prompt = cfg.search_space["prompts"]["answer_prompt_seed"]
        note = "baseline"
        if source == "best":
            latest = next((d for d in sorted(runs_dir.glob("*optimize"), reverse=True)
                           if (d / "best_configs.json").exists()), None)
            best = _read_json(latest / "best_configs.json") if latest else None
            if best and best.get("arms"):
                arm_id = arm if arm in best["arms"] else next(iter(best["arms"]))
                bc = best["arms"][arm_id]["best_config"]
                params = params_from_dict(bc["retrieval"])
                answer_prompt = bc["answer_prompt"]
                note = f"optimized:{arm_id} (from {latest.name})"
        rag_cache[key] = (factory.build(params, answer_prompt), note)
        return rag_cache[key]

    @app.post("/api/chat")
    def chat(req: ChatReq):
        rag, note = _get_rag(req.source, req.arm)
        timing = TimingLog()
        rag.timing = timing
        result = rag.ask(req.question)
        return {
            "answer": result.answer,
            "config_note": note,
            "sources": [{"chunk_id": r.chunk.chunk_id, "doc_id": r.chunk.doc_id,
                         "score": round(r.score, 4), "text": r.chunk.text[:400]}
                        for r in result.sources],
            "timings": timing.records,
            "total_seconds": timing.total(),
        }

    @app.get("/")
    def index():
        return FileResponse(_HERE / "index.html")

    return app
