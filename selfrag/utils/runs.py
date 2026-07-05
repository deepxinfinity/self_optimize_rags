"""Timestamped run directories and JSON logging under runs/."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


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


def append_jsonl(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_run_meta(run_dir: str | Path, corpus_path: str, config_path: str, kind: str,
                   arm: str | None = None, split: str | None = None) -> None:
    """Written at the start of every run so the dashboard can attribute + label it (and show
    live/in-progress runs), even before any results file exists."""
    import os
    write_json(Path(run_dir) / "run_meta.json", {
        "kind": kind,
        "corpus_path": corpus_path,
        "config": config_path,
        "arm": arm,
        "split": split,
        "pid": os.getpid(),   # lets the UI tell "running" from "aborted" reliably
        "started_at": datetime.now().isoformat(timespec="seconds"),
    })
