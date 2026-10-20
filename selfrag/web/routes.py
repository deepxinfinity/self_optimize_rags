"""REST API for projects, ingestion, corpus reports and optimize jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..config import Config
from . import ingest, jobs
from .ingest import IngestError, IngestLimits
from .projects import ProjectError, ProjectStore
from .report import corpus_report


class ConfigChoices(BaseModel):
    generator: Optional[str] = None
    judge: Optional[str] = None
    arm: Optional[dict[str, Any]] = None
    n_pairs: Optional[int] = None
    stage1_trials: Optional[int] = None
    stage2_trials: Optional[int] = None
    overrides: list[str] = []


class OptimizeReq(BaseModel):
    force_regen: bool = False


def _err(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def build_router(cfg: Config, base_config_path: str) -> APIRouter:
    router = APIRouter(prefix="/api")
    store = ProjectStore(cfg.runtime.projects_dir)
    limits = IngestLimits.from_cfg(cfg.runtime)
    projects_root = Path(cfg.runtime.projects_dir)

    def _project_or_404(pid: str):
        meta = store.get(pid)
        if meta is None:
            return None, _err(404, f"unknown project: {pid}")
        return meta, None


    @router.get("/projects")
    def list_projects():
        out = []
        for meta in store.list():
            pdir = store.path(meta["id"])
            corpus = pdir / "corpus"
            n_docs = sum(1 for p in corpus.rglob("*") if p.is_file()
                         and p.suffix in (".txt", ".md")) if corpus.exists() else 0
            running = any(j["status"] == "running" for j in jobs.list_jobs(pdir))
            out.append({**meta, "n_docs": n_docs, "has_config": (pdir / "config.yaml").exists(),
                        "job_running": running})
        return {"projects": out}

    @router.post("/projects", status_code=201)
    def create_project(body: dict[str, Any]):
        try:
            return store.create(str(body.get("name", "")))
        except ProjectError as e:
            return _err(422, str(e))

    @router.get("/projects/{pid}")
    def get_project(pid: str):
        meta, err = _project_or_404(pid)
        if err:
            return err
        pdir = store.path(pid)
        return {**meta,
                "has_config": (pdir / "config.yaml").exists(),
                "has_real_questions": (pdir / "real_questions.jsonl").exists(),
                "jobs": jobs.list_jobs(pdir)}

    @router.delete("/projects/{pid}")
    def delete_project(pid: str):
        for job in jobs.list_jobs(store.path(pid)):
            if job["status"] == "running":
                return _err(409, "a job is running, cancel it before deleting the project")
        if not store.delete(pid):
            return _err(404, f"unknown project: {pid}")
        return {"deleted": pid}


    @router.post("/projects/{pid}/data")
    async def add_data(pid: str,
                       files: Optional[list[UploadFile]] = File(None),
                       url: Optional[str] = Form(None),
                       hf: Optional[str] = Form(None),
                       text_column: Optional[str] = Form(None),
                       max_rows: int = Form(300)):
        meta, err = _project_or_404(pid)
        if err:
            return err
        pdir = store.path(pid)
        entries: list[dict[str, Any]] = []
        try:
            for f in files or []:
                data = await f.read()
                entries.append(ingest.ingest_bytes(pdir, f.filename or "upload.txt", data, limits))
            if url:
                entries.extend(ingest.ingest_zip_url(pdir, url, limits))
            if hf:
                entries.append(ingest.ingest_hf(pdir, hf, limits,
                                                text_column=text_column, max_rows=max_rows))
        except IngestError as e:
            return _err(422, str(e))
        if not entries:
            return _err(422, "provide files, a url, or an hf dataset spec")
        store.add_source_event(pid, {"entries": entries})
        return {"ingested": entries, "report": corpus_report(pdir)}

    @router.post("/projects/{pid}/real-questions")
    async def add_real_questions(pid: str, file: UploadFile = File(...)):
        meta, err = _project_or_404(pid)
        if err:
            return err
        raw = (await file.read()).decode("utf-8", errors="replace")
        pairs = []
        for i, line in enumerate(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return _err(422, f"line {i + 1}: not valid JSON")
            if not obj.get("question") or not obj.get("gold_doc_id"):
                return _err(422, f"line {i + 1}: each line needs 'question' and 'gold_doc_id'")
            pairs.append(obj)
        if len(pairs) < 10:
            return _err(422, f"need at least 10 real questions for a meaningful check, got {len(pairs)}")
        out = store.path(pid) / "real_questions.jsonl"
        out.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n")
        store.add_source_event(pid, {"entries": [{"source": "real_questions.jsonl",
                                                  "origin": "upload", "n_questions": len(pairs)}]})
        return {"real_questions": len(pairs),
                "note": "re-save the config so it picks up real_questions_path"}

    @router.get("/projects/{pid}/corpus")
    def get_corpus_report(pid: str):
        meta, err = _project_or_404(pid)
        if err:
            return err
        return corpus_report(store.path(pid))


    @router.put("/projects/{pid}/config")
    def put_config(pid: str, choices: ConfigChoices):
        meta, err = _project_or_404(pid)
        if err:
            return err
        try:
            path = store.write_config(pid, base_config_path, choices.model_dump())
        except (ProjectError, ValueError) as e:
            return _err(422, f"config rejected: {e}")
        effective = yaml.safe_load(path.read_text())
        return {"config_path": str(path),
                "models": {"generator": effective["models"]["generator"]["name"],
                           "judge": effective["models"]["judge"]["name"],
                           "arms": [a["id"] for a in effective["models"]["eval_generators"]]},
                "has_real_questions": bool(effective["dataset"].get("real_questions_path"))}

    @router.get("/projects/{pid}/config")
    def get_config(pid: str):
        meta, err = _project_or_404(pid)
        if err:
            return err
        path = store.path(pid) / "config.yaml"
        if not path.exists():
            return _err(404, "no config saved yet")
        return yaml.safe_load(path.read_text())


    @router.post("/projects/{pid}/optimize", status_code=201)
    def optimize(pid: str, req: Optional[OptimizeReq] = None):
        meta, err = _project_or_404(pid)
        if err:
            return err
        pdir = store.path(pid)
        try:
            return jobs.start_job(projects_root, pdir, pdir / "config.yaml",
                                  force_regen=bool(req and req.force_regen))
        except jobs.JobError as e:
            return _err(409, str(e))

    @router.get("/projects/{pid}/jobs")
    def list_project_jobs(pid: str):
        meta, err = _project_or_404(pid)
        if err:
            return err
        return {"jobs": jobs.list_jobs(store.path(pid))}

    @router.get("/projects/{pid}/jobs/{job_id}")
    def get_project_job(pid: str, job_id: str, log_lines: int = 60):
        meta, err = _project_or_404(pid)
        if err:
            return err
        job = jobs.get_job(store.path(pid), job_id)
        if job is None:
            return _err(404, f"unknown job: {job_id}")
        job["log_tail"] = jobs.tail_log(job.get("log", ""), n=log_lines)
        return job

    @router.delete("/projects/{pid}/jobs/{job_id}")
    def cancel_project_job(pid: str, job_id: str):
        meta, err = _project_or_404(pid)
        if err:
            return err
        try:
            return jobs.cancel_job(store.path(pid), job_id)
        except jobs.JobError as e:
            return _err(404, str(e))

    return router
