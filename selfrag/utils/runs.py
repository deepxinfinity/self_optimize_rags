from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def new_run_dir(out_dir: str | Path, tag: str = "") -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{ts}-{tag}" if tag else ts
    path = Path(out_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def write_jsonl(path: str | Path, rows: list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # process exists, just not ours to signal
    return True


def optimize_progress(run_dir: Path) -> dict[str, dict[str, int]]:
    """{arm: {stage1: n, stage2: n}} trial counts for a run that may still be going."""
    prog: dict[str, dict[str, int]] = {}
    for f in run_dir.glob("*_trials.jsonl"):
        stem = f.stem
        for st in ("stage1", "stage2"):
            suffix = f"_{st}_trials"
            if stem.endswith(suffix):
                arm = stem[: -len(suffix)]
                prog.setdefault(arm, {})[st] = sum(1 for _ in f.open())
    return prog


def write_run_meta(run_dir: str | Path, corpus_path: str, config_path: str, kind: str,
                   arm: str | None = None, split: str | None = None,
                   overrides: list[str] | None = None) -> None:
    # overrides are recorded because the run is not reproducible from the config file alone
    write_json(Path(run_dir) / "run_meta.json", {
        "kind": kind,
        "corpus_path": corpus_path,
        "config": config_path,
        "overrides": list(overrides or []),
        "arm": arm,
        "split": split,
        "pid": os.getpid(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    })
