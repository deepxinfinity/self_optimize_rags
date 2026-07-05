"""Config loading, --set overrides, and validation.

Single source of truth: every model, path, hyperparameter range, and metric choice comes from a
YAML config (configs/*.yaml). CLI flags may *override* keys via ``--set a.b.c=value`` but defaults
live in the file. Nothing in the rest of the codebase should hardcode a model, path, or range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, field_validator


class _Base(BaseModel):
    # extra="allow": keep (and expose) keys we don't explicitly model — the config carries
    # optional per-model fields (temperature, max_tokens, api, ...) that vary by provider.
    model_config = ConfigDict(extra="allow")


class ModelSpec(_Base):
    provider: str
    name: str


class EvalArm(_Base):
    id: str
    provider: str
    name: str


class ModelsCfg(_Base):
    generator: ModelSpec
    embedding: ModelSpec
    reranker: ModelSpec
    judge: ModelSpec
    eval_generators: list[EvalArm]


class DatasetCfg(_Base):
    corpus_path: str
    corpus_glob: str = "**/*.{txt,md}"
    real_questions_path: Optional[str] = None


class SplitsCfg(_Base):
    train: float
    val: float
    test: float
    seed: int = 42

    @field_validator("test")
    @classmethod
    def _sums_to_one(cls, v: float, info):
        total = info.data.get("train", 0) + info.data.get("val", 0) + v
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"splits train+val+test must sum to 1.0, got {total}")
        return v


class RuntimeCfg(_Base):
    cache_dir: str = "./.cache/api"
    out_dir: str = "./runs"
    log_every_trial: bool = True


class Config(_Base):
    models: ModelsCfg
    dataset: DatasetCfg
    splits: SplitsCfg
    search_space: dict[str, Any]
    # Kept as loose dicts for now; interpreted by later increments (evalgen/optimize).
    evalgen: dict[str, Any] = {}
    optimization: dict[str, Any] = {}
    runtime: RuntimeCfg = RuntimeCfg()

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


def _parse_scalar(raw: str) -> Any:
    """Type a --set right-hand side the way YAML would (int/float/bool/null), else keep string.

    ``top_k=5`` -> int, ``use_reranker=true`` -> bool, ``name=llama3.2:3b`` -> str (a colon not
    followed by a space stays a plain scalar in YAML), ``corpus_path=./data`` -> str.
    """
    try:
        val = yaml.safe_load(raw)
        # yaml turns an empty string into None; preserve the user's literal empty string instead.
        return raw if val is None and raw != "null" else val
    except yaml.YAMLError:
        return raw


def _apply_overrides(data: dict[str, Any], overrides: list[str]) -> None:
    """Apply dotted ``a.b.c=value`` overrides onto the raw config dict, in place."""
    for ov in overrides:
        key, sep, val = ov.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got: {ov!r}")
        keys = key.strip().split(".")
        node: Any = data
        for k in keys[:-1]:
            nxt = node.get(k)
            if not isinstance(nxt, dict):
                nxt = {}
                node[k] = nxt
            node = nxt
        node[keys[-1]] = _parse_scalar(val)


def load_config(path: str | Path, overrides: Optional[list[str]] = None) -> Config:
    """Load YAML, apply --set overrides, validate into a frozen-ish typed Config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if overrides:
        _apply_overrides(raw, overrides)
    return Config(**raw)
