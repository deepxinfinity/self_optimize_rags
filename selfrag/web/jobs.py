"""Optimize jobs as subprocesses: `python -m selfrag pipeline --config projects/<id>/config.yaml`.

Each job runs in its own session so it can be killed as a group. Output goes to
projects/<id>/logs/, progress is read from the run directory the CLI writes. One job per project
and GLOBAL_MAX_JOBS overall: two local LLM jobs do not fit in 16 GB.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..utils.runs import optimize_progress, pid_alive

GLOBAL_MAX_JOBS = 1

# Popen handles for jobs this server started; after a restart we only have the pid.
_PROCS: dict[str, subprocess.Popen] = {}


class JobError(RuntimeError):
    pass


def _jobs_dir(project_dir: Path) -> Path:
    d = project_dir / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_record(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_record(project_dir: Path, record: dict[str, Any]) -> None:
    (_jobs_dir(project_dir) / f"{record['job_id']}.json").write_text(
        json.dumps(record, indent=2))


def list_jobs(project_dir: Path) -> list[dict[str, Any]]:
    records = []
    for p in sorted(_jobs_dir(project_dir).glob("*.json"), reverse=True):
        rec = _read_record(p)
        if rec:
            records.append(job_status(project_dir, rec))
    return records


def get_job(project_dir: Path, job_id: str) -> Optional[dict[str, Any]]:
    rec = _read_record(_jobs_dir(project_dir) / f"{job_id}.json")
    return job_status(project_dir, rec) if rec else None


def _running_jobs(projects_root: Path) -> list[dict[str, Any]]:
    running = []
    for jobs_dir in projects_root.glob("*/jobs"):
        for p in jobs_dir.glob("*.json"):
            rec = _read_record(p)
            if rec and _is_alive(rec):
                running.append(rec)
    return running


def _is_alive(record: dict[str, Any]) -> bool:
    proc = _PROCS.get(record["job_id"])
    if proc is not None:
        return proc.poll() is None
    return pid_alive(record.get("pid"))


def start_job(projects_root: Path, project_dir: Path, config_path: Path,
              force_regen: bool = False, cmd: Optional[list[str]] = None) -> dict[str, Any]:
    if not config_path.exists():
        raise JobError("project has no config.yaml, save the configuration first")
    for rec in _running_jobs(projects_root):
        if Path(rec["project_dir"]) == project_dir:
            raise JobError(f"project already has a running job: {rec['job_id']}")
    if len(_running_jobs(projects_root)) >= GLOBAL_MAX_JOBS:
        raise JobError(f"job limit reached ({GLOBAL_MAX_JOBS} concurrent), try again later")

    job_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_dir = project_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"optimize-{job_id}.log"

    if cmd is None:
        cmd = [sys.executable, "-m", "selfrag", "pipeline", "--config", str(config_path)]
        if force_regen:
            cmd.append("--force")

    with log_path.open("w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                cwd=str(Path.cwd()), start_new_session=True)
    record = {
        "job_id": job_id,
        "project_dir": str(project_dir),
        "pid": proc.pid,
        "cmd": cmd,
        "log": str(log_path),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    _PROCS[job_id] = proc
    _write_record(project_dir, record)
    return job_status(project_dir, record)


def cancel_job(project_dir: Path, job_id: str) -> dict[str, Any]:
    rec = _read_record(_jobs_dir(project_dir) / f"{job_id}.json")
    if rec is None:
        raise JobError(f"unknown job: {job_id}")
    if _is_alive(rec):
        try:
            os.killpg(rec["pid"], signal.SIGTERM)   # own session, so pgid == pid
        except ProcessLookupError:
            pass
        rec["cancelled_at"] = datetime.now().isoformat(timespec="seconds")
        _write_record(project_dir, rec)
    return job_status(project_dir, rec)


def _newest_run(project_dir: Path, job_id: str) -> Optional[Path]:
    runs_dir = project_dir / "runs"
    if not runs_dir.exists():
        return None
    # run dirs and job ids share the YYYYMMDD-HHMMSS prefix
    candidates = [d for d in runs_dir.iterdir()
                  if d.is_dir() and "-".join(d.name.split("-")[:2]) >= job_id]
    return max(candidates, default=None, key=lambda d: d.name)


def tail_log(path: str | Path, n: int = 60) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    return p.read_text(errors="replace").splitlines()[-n:]


def job_status(project_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    alive = _is_alive(record)
    run_dir = _newest_run(project_dir, record["job_id"])
    done = run_dir is not None and (run_dir / "best_configs.json").exists()

    proc = _PROCS.get(record["job_id"])
    exit_code = proc.poll() if proc is not None else None

    if record.get("cancelled_at"):
        out["status"] = "cancelled" if not alive else "cancelling"
    elif alive:
        out["status"] = "running"
    elif done:
        out["status"] = "complete"
    else:
        out["status"] = "failed"
        out["exit_code"] = exit_code
        out["error_tail"] = tail_log(record.get("log", ""), n=15)

    if run_dir is not None:
        out["run"] = run_dir.name
        out["progress"] = optimize_progress(run_dir)
        if done:
            try:
                best = json.loads((run_dir / "best_configs.json").read_text())
                out["headline_metric"] = best.get("headline_metric")
                out["has_real_heldout"] = best.get("has_real_heldout")
                out["scores"] = {a: v.get("scores", {}) for a, v in best.get("arms", {}).items()}
            except (OSError, json.JSONDecodeError):
                pass
    elif alive:
        out["phase"] = "generating eval set"
    return out
