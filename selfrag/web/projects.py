"""Project workspaces: projects/<id>/ holds meta.json, config.yaml, corpus/, evalsets/, runs/.

config.yaml is the base config with the project's paths and the user's choices applied, validated
through load_config, so `python -m selfrag pipeline --config projects/<id>/config.yaml` works as is.
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from ..config import _apply_overrides, load_config

_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")[:40]
    return slug or "project"


class ProjectError(ValueError):
    pass


class ProjectStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)


    def create(self, name: str) -> dict[str, Any]:
        if not name or not name.strip():
            raise ProjectError("project name is required")
        pid = slugify(name)
        if (self.root / pid).exists():
            pid = f"{pid}-{secrets.token_hex(3)}"
        pdir = self.root / pid
        for sub in ("corpus", "evalsets", "runs", "logs", "jobs"):
            (pdir / sub).mkdir(parents=True, exist_ok=True)
        meta = {
            "id": pid,
            "name": name.strip(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sources": [],
        }
        self._write_meta(pid, meta)
        return meta

    def list(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        out = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and (d / "meta.json").exists():
                meta = self.get(d.name)
                if meta:
                    out.append(meta)
        return out

    def get(self, pid: str) -> Optional[dict[str, Any]]:
        p = self.root / pid / "meta.json"
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def path(self, pid: str) -> Path:
        return self.root / pid

    def delete(self, pid: str) -> bool:
        pdir = self.root / pid
        if not (pdir / "meta.json").exists():
            return False
        shutil.rmtree(pdir)
        return True


    def _write_meta(self, pid: str, meta: dict[str, Any]) -> None:
        (self.root / pid / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    def add_source_event(self, pid: str, event: dict[str, Any]) -> None:
        meta = self.get(pid)
        if meta is None:
            raise ProjectError(f"unknown project: {pid}")
        event["at"] = datetime.now().isoformat(timespec="seconds")
        meta.setdefault("sources", []).append(event)
        self._write_meta(pid, meta)


    def write_config(self, pid: str, base_config_path: str | Path,
                     choices: Optional[dict[str, Any]] = None) -> Path:
        """choices: generator, judge, arm, n_pairs, stage1_trials, stage2_trials, overrides."""
        pdir = self.path(pid)
        if not (pdir / "meta.json").exists():
            raise ProjectError(f"unknown project: {pid}")
        choices = choices or {}

        raw = yaml.safe_load(Path(base_config_path).read_text()) or {}

        # absolute so the job subprocess doesn't depend on its cwd
        raw.setdefault("dataset", {})
        raw["dataset"]["corpus_path"] = str((pdir / "corpus").resolve())
        raw["dataset"]["corpus_glob"] = "**/*.{txt,md}"
        real_q = pdir / "real_questions.jsonl"
        raw["dataset"]["real_questions_path"] = str(real_q.resolve()) if real_q.exists() else None
        raw.setdefault("evalgen", {})["out_dir"] = str((pdir / "evalsets").resolve())
        raw.setdefault("runtime", {})["out_dir"] = str((pdir / "runs").resolve())
        if raw["runtime"].get("cache_dir"):
            raw["runtime"]["cache_dir"] = str(Path(raw["runtime"]["cache_dir"]).resolve())

        if choices.get("generator"):
            raw["models"]["generator"]["name"] = choices["generator"]
        if choices.get("judge"):
            raw["models"]["judge"]["name"] = choices["judge"]
        if choices.get("arm"):
            arm = dict(choices["arm"])
            arm.setdefault("id", "user_arm")
            arm.setdefault("provider", "ollama")
            if not arm.get("name"):
                raise ProjectError("arm.name (the eval-generator model) is required")
            raw["models"]["eval_generators"] = [arm]
        elif choices.get("single_arm", True) and len(raw["models"].get("eval_generators", [])) > 1:
            # projects run a single arm; two arms is the research setup
            raw["models"]["eval_generators"] = raw["models"]["eval_generators"][:1]
        if choices.get("n_pairs"):
            raw.setdefault("evalgen", {})["n_pairs_per_arm"] = int(choices["n_pairs"])
        if choices.get("stage1_trials"):
            raw.setdefault("optimization", {}).setdefault("stage1_retrieval", {})["n_trials"] = \
                int(choices["stage1_trials"])
        if choices.get("stage2_trials"):
            raw.setdefault("optimization", {}).setdefault("stage2_prompts", {})["n_trials"] = \
                int(choices["stage2_trials"])
        if choices.get("overrides"):
            _apply_overrides(raw, list(choices["overrides"]))

        out = pdir / "config.yaml"
        out.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
        load_config(out)
        return out
