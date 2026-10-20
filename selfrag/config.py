from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, field_validator


class _Base(BaseModel):
    # model specs carry provider-specific extras (temperature, num_ctx, api, ...)
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
    embedding_cache_dir: Optional[str] = "./.cache/embeddings"
    out_dir: str = "./runs"
    log_every_trial: bool = True
    projects_dir: str = "./projects"
    max_upload_mb: int = 25
    max_project_mb: int = 200


class Config(_Base):
    models: ModelsCfg
    dataset: DatasetCfg
    splits: SplitsCfg
    search_space: dict[str, Any]
    baseline: dict[str, Any] = {}
    retrieval: dict[str, Any] = {}
    evalgen: dict[str, Any] = {}
    optimization: dict[str, Any] = {}
    runtime: RuntimeCfg = RuntimeCfg()

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


def _parse_scalar(raw: str) -> Any:
    # parse like YAML would, but keep a literal empty string as ""
    try:
        val = yaml.safe_load(raw)
        return raw if val is None and raw != "null" else val
    except yaml.YAMLError:
        return raw


def _apply_overrides(data: dict[str, Any], overrides: list[str]) -> None:
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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if overrides:
        _apply_overrides(raw, overrides)
    return Config(**raw)
